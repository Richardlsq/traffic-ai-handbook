from math import isfinite
from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, checkpoint_name=None):
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_name = checkpoint_name
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_epoch = None

    def __call__(self, val_loss, model_dict, model, epoch, save_path):
        if not isfinite(val_loss):
            raise ValueError(f'Non-finite validation loss at epoch {epoch}: {val_loss}')
        if val_loss < self.val_loss_min:
            self.save_checkpoint(val_loss, model_dict, model, epoch, save_path)
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience

    def save_checkpoint(self, val_loss, model_dict, model, epoch, save_path):
        filename = self.checkpoint_name or f'model_dict_checkpoint_{epoch}_{val_loss:.8f}.pth'
        location = Path(save_path) / filename
        temporary = location.with_suffix('.tmp')
        torch.save(model_dict, temporary)
        temporary.replace(location)
        if self.verbose:
            print(f'Validation MSE improved from {self.val_loss_min:.8f} to {val_loss:.8f}', flush=True)
        self.val_loss_min = val_loss
        self.best_epoch = epoch
