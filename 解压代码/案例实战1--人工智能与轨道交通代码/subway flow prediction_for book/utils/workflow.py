import argparse
import csv
from datetime import datetime
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path

import numpy as np
import torch

from data.get_dataloader import get_inflow_dataloader, get_outflow_dataloader
from model.main_model import Model
from utils.earlystopping import EarlyStopping
from utils.metrics import Metrics
from utils.utils import GetLaplacian


PROJECT = Path(__file__).resolve().parents[1]
NODES = 276
ADJACENCY = PROJECT / 'data' / 'adjacency.csv'
INFLOW_FILE = PROJECT / 'data' / 'in_15min.csv'
OUTFLOW_FILE = PROJECT / 'data' / 'out_15min.csv'
DEFAULT_EPOCHS = 1000


def _device(name):
    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    return device


def _config(holdout_days, batch_size):
    return {
        'time_interval': 15, 'time_lag': 10, 'tg_in_one_day': 72,
        'forecast_day_number': holdout_days, 'pre_len': 1, 'batch_size': batch_size,
    }


def _setup(config, device, *, inflow_only=False):
    *inflow, maximum, minimum = get_inflow_dataloader(**config)
    if inflow_only:
        splits = [(loader,) for loader in inflow]
    else:
        *outflow, _, _ = get_outflow_dataloader(**config)
        splits = list(zip(inflow, outflow, strict=True))
        for flow_in, flow_out in splits:
            if not np.array_equal(flow_in.dataset.target_indices, flow_out.dataset.target_indices):
                raise ValueError('Inflow and outflow windows cover different time indices')
    adjacency = GetLaplacian(np.loadtxt(ADJACENCY, delimiter=',')).get_normalized_adj(NODES)
    graph = torch.as_tensor(adjacency, device=device)
    model = Model(config['time_lag'], config['pre_len'], NODES, device).to(device)
    return splits, (maximum, minimum), graph, model


def _data_hashes(*, inflow_only=False):
    files = (ADJACENCY, INFLOW_FILE) if inflow_only else (ADJACENCY, INFLOW_FILE, OUTFLOW_FILE)
    fingerprints = {}
    for path in files:
        with path.open('rb') as stream:
            fingerprints[path.name] = file_digest(stream, 'sha256').hexdigest()
    return fingerprints


def _batches(loaders, device):
    for batches in zip(*loaders, strict=True):
        inflow = batches[0]
        outflow = batches[1][0].to(device) if len(batches) == 2 else None
        yield (inflow[0].to(device), outflow), inflow[1].to(device), inflow[2:]


def _train_epoch(model, loaders, graph, optimizer):
    model.train()
    sum_squared = torch.zeros((), device=graph.device)
    count = 0
    for features, target, _ in _batches(loaders, graph.device):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(*features, graph)
        error = torch.nn.functional.mse_loss(prediction, target, reduction='sum')
        (error / target.numel()).backward()
        optimizer.step()
        sum_squared += error.detach()
        count += target.numel()
    return sum_squared.item() / count


@torch.inference_mode()
def _validation_mse(model, validation_loaders, graph):
    model.eval()
    sum_squared = torch.zeros((), device=graph.device, dtype=torch.float64)
    count = 0
    for features, target, _ in _batches(validation_loaders, graph.device):
        prediction = model(*features, graph)
        sum_squared += (prediction.to(torch.float64) - target).square().sum()
        count += target.numel()
    return sum_squared.item() / count


def _metrics(actual, forecast):
    def by_node(values):
        return values.transpose(0, 2, 1).reshape(-1, values.shape[1])

    rmse, r2, mae, wmape = Metrics(by_node(actual), by_node(forecast)).evaluate_performance()
    return {'rmse': rmse, 'mae': mae, 'r2': r2 if isfinite(r2) else None,
            'wmape': wmape if isfinite(wmape) else None}


@torch.inference_mode()
def _evaluate(model, loaders, graph, scaling, output, checkpoint_epoch):
    model.eval()
    maximum, minimum = scaling
    span = maximum - minimum or 1.0
    predictions, observations, reference = [], [], []
    sum_squared = torch.zeros((), device=graph.device, dtype=torch.float64)
    count = 0
    for features, target, (original,) in _batches(loaders, graph.device):
        prediction = model(*features, graph)
        sum_squared += (prediction.to(torch.float64) - target).square().sum()
        count += target.numel()
        predictions.append(prediction.cpu().numpy().astype(np.float64) * span + minimum)
        observations.append(original.numpy())
        last = features[0][:, :, -1:].cpu().numpy().astype(np.float64) * span + minimum
        reference.append(np.broadcast_to(last, original.shape))
    predicted = np.maximum(np.concatenate(predictions), 0.0)
    actual = np.concatenate(observations)
    persistence = np.concatenate(reference)
    indices = loaders[0].dataset.target_indices
    result = {
        'checkpoint_epoch': checkpoint_epoch, 'split_metadata_verified': checkpoint_epoch is not None,
        'inputs': ['inflow'] if len(loaders) == 1 else ['inflow', 'outflow'],
        'samples': len(indices), 'first_target_index': int(indices[0]), 'last_target_index': int(indices[-1]),
        'normalized_mse': sum_squared.item() / count, 'model': _metrics(actual, predicted),
        'last_observation': _metrics(actual, persistence),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'predictions.npz', target_index=indices, actual=actual,
                        predicted=predicted, last_observation=persistence)
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def train_main():
    parser = argparse.ArgumentParser(description='Train on chronological data and save the best validation checkpoint.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--holdout-days', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--inflow-only', action='store_true',
                        help='Mask the outflow channel for an otherwise matched ablation.')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.holdout_days) < 1 or not isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs, patience, lr, batch size and holdout days must be positive')

    torch.manual_seed(args.seed)
    device = _device(args.device)
    config = _config(args.holdout_days, args.batch_size)
    (train, validation, test), scaling, graph, model = _setup(config, device, inflow_only=args.inflow_only)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = args.output_dir or PROJECT / 'outputs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    stopper = EarlyStopping(patience=args.patience, checkpoint_name='best.pth')
    fingerprints = _data_hashes(inflow_only=args.inflow_only)
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(output / 'tensorboard'))

    splits = {}
    for name, loaders in zip(('train', 'validation', 'test'), (train, validation, test), strict=True):
        indices = loaders[0].dataset.target_indices
        splits[name] = {'samples': len(indices), 'start': int(indices[0]), 'end': int(indices[-1])}
    print(f'Device: {device}. Inflow only: {args.inflow_only}. Splits: {json.dumps(splits)}', flush=True)
    try:
        with (output / 'training.csv').open('w', newline='', encoding='utf-8') as stream:
            history = csv.writer(stream)
            history.writerow(('epoch', 'train_mse', 'validation_mse'))
            for epoch in range(1, args.epochs + 1):
                train_mse = _train_epoch(model, train, graph, optimizer)
                val_mse = _validation_mse(model, validation, graph)
                payload = {
                    'model_state_dict': model.state_dict(), 'config': config, 'splits': splits, 'data_sha256': fingerprints,
                    'graph_project': 'metro', 'seed': args.seed, 'epoch': epoch, 'validation_mse': val_mse,
                    'inflow_only': args.inflow_only,
                    'maximum': scaling[0], 'minimum': scaling[1],
                }
                stopper(val_mse, payload, model, epoch, output)
                history.writerow((epoch, train_mse, val_mse))
                stream.flush()
                if writer:
                    writer.add_scalar('loss_train', train_mse, epoch)
                    writer.add_scalar('loss_validation', val_mse, epoch)
                print(f'Epoch {epoch}: train MSE={train_mse:.8f}, validation MSE={val_mse:.8f}', flush=True)
                if stopper.early_stop:
                    break
    finally:
        if writer:
            writer.close()
    print(f'Best epoch: {stopper.best_epoch}; validation MSE: {stopper.val_loss_min:.8f}', flush=True)
    print(f'Checkpoint: {output / "best.pth"}', flush=True)


def predict_main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on its chronological test split.')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = _device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('graph_project') != 'metro':
        parser.error('Choose a checkpoint produced by this project')
    inflow_only = checkpoint.get('inflow_only', False)
    if checkpoint['data_sha256'] != _data_hashes(inflow_only=inflow_only):
        raise ValueError('Data or graph files differ from the saved training run')
    (_, validation, test), scaling, graph, model = _setup(checkpoint['config'], device, inflow_only=inflow_only)
    if scaling != (checkpoint['maximum'], checkpoint['minimum']):
        raise ValueError('Data scaling differs from the saved training run')
    test_indices = test[0].dataset.target_indices
    saved_test = checkpoint['splits']['test']
    if (len(test_indices), test_indices[0], test_indices[-1]) != (saved_test['samples'], saved_test['start'], saved_test['end']):
        raise ValueError('The test split differs from the saved training run')
    model.load_state_dict(checkpoint['model_state_dict'])
    val_mse = _validation_mse(model, validation, graph)
    print(f'Checkpoint epoch {checkpoint["epoch"]}: saved validation MSE={checkpoint["validation_mse"]:.8f}, '
          f'recomputed validation MSE={val_mse:.8f}', flush=True)
    _evaluate(model, test, graph, scaling, args.output_dir or args.checkpoint.parent / 'test', checkpoint['epoch'])
