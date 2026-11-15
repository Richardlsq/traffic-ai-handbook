import argparse
from pathlib import Path
import random
from random import choice, choices, randint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms.community.modularity_max import \
    greedy_modularity_communities

def draw_graph(
    graph: nx.Graph,
    color_map: List[str] = None,
    positions: Dict[int, Tuple[float, float]] = None,
    axis_labels: Tuple[str, str] = None,
    with_edges=True,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 13))

    nx.draw(
        graph,
        nx.spring_layout(graph) if positions is None else positions,
        node_size=400,
        node_color=color_map,
        with_labels=True,
        ax=ax,
    )

    if axis_labels is not None:
        plt.axis('on')
        ax.tick_params(left=True, bottom=True,
                       labelleft=True, labelbottom=True)
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])


def create_graph() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, nx.Graph]:
    number_of_samples = 200
    probability_of_edge = 0.03
    seed = 13

    random.seed(seed)

    feature_range = {
        0: {
            0: (1, 3),
            1: (1, 3),
            2: (1, 10),
            3: (1, 10),
            4: (7, 10),
            5: (4, 6),
        },
        1: {
            0: (1, 5),
            1: (6, 10),
            2: (1, 3),
            3: (1, 5),
            4: (5, 10),
            5: (1, 3),
        },
        2: {
            0: (4, 7),
            1: (1, 3),
            2: (1, 10),
            3: (4, 6),
            4: (1, 3),
            5: (1, 10),
        },
        3: {
            0: (1, 2),
            1: (2, 6),
            2: (8, 10),
            3: (8, 10),
            4: (8, 10),
            5: (4, 10),
        },
        4: {
            0: (3, 8),
            1: (4, 6),
            2: (2, 5),
            3: (1, 3),
            4: (1, 4),
            5: (1, 5),
        },
        5: {
            0: (4, 7),
            1: (5, 8),
            2: (5, 8),
            3: (7, 10),
            4: (5, 10),
            5: (3, 6),
        },
        6: {
            0: (7, 10),
            1: (9, 10),
            2: (1, 4),
            3: (1, 10),
            4: (3, 10),
            5: (6, 10),
        },
        7: {
            0: (3, 8),
            1: (7, 10),
            2: (8, 10),
            3: (5, 10),
            4: (6, 10),
            5: (1, 10),
        },
    }

    graph = nx.fast_gnp_random_graph(
        number_of_samples, probability_of_edge, seed=seed)

    communities = list(greedy_modularity_communities(graph))

    features = [[] for _ in range(number_of_samples)]
    labels = [[] for _ in range(number_of_samples)]

    for community in range(len(communities)):
        community_features = feature_range[community]
        label = 0 if community in [0, 1, 2, 4] else 1

        for v in communities[community]:
            features[v] = [
                randint(community_features[0][0], community_features[0][1]),
                randint(community_features[1][0], community_features[1][1]),
                randint(community_features[2][0], community_features[2][1]),
                randint(community_features[3][0], community_features[3][1]),
                randint(community_features[4][0], community_features[4][1]),
                randint(community_features[5][0], community_features[5][1]),
            ]
            labels[v] = choices(
                [0, 1] if label == 0 else [1, 0], weights=[95, 5])[0]

    indexes = list(range(number_of_samples))
    train_indexes = []

    while len(set(train_indexes)) <= 50:
        train_indexes.append(choice(indexes))

    train_indexes = list(set(train_indexes))

    features = torch.Tensor(features) / 10

    train_labels = torch.tensor([[labels[i], i]
                                 for i in train_indexes], dtype=torch.long)
    validation_labels = torch.tensor(labels, dtype=torch.long)

    adjacency_matrix = torch.from_numpy(nx.to_numpy_array(graph))

    return features, train_labels, validation_labels, adjacency_matrix, graph

class GraphConv(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
    ):
        super(GraphConv, self).__init__()
        self._input_features = input_features
        self._output_features = output_features
        self.weight = nn.Parameter(
            torch.Tensor(input_features, output_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return adjacency_matrix @ inputs @ self.weight + self.bias

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self._graph_conv_1 = GraphConv(6, 4)
        self._activation_1 = nn.ReLU()
        self._graph_conv_2 = GraphConv(4, 2)
        self._activation_2 = nn.LogSoftmax(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> torch.Tensor:
        x = self._graph_conv_1(inputs, adjacency_matrix)
        x = self._activation_1(x)
        x = self._graph_conv_2(x, adjacency_matrix)
        x = self._activation_2(x)

        return x


def normalize_adjacency(adjacency):
    with_self_loops = adjacency + torch.eye(len(adjacency), dtype=adjacency.dtype, device=adjacency.device)
    inv_degree = with_self_loops.sum(dim=1).rsqrt()
    return inv_degree[:, None] * with_self_loops * inv_degree[None, :]


def train(model, features, train_labels, labels, adjacency, epochs):
    optimizer = torch.optim.Adam(model.parameters())
    train_nodes = train_labels[:, 1]
    validation = torch.ones(len(labels), dtype=torch.bool)
    validation[train_nodes] = False
    validation_count = int(validation.sum())
    print(f'Train nodes: {len(train_nodes)}, validation nodes: {validation_count}', flush=True)
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(features, adjacency)
        train_output = output[train_nodes]
        train_loss = F.nll_loss(train_output, train_labels[:, 0])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(features, adjacency)
            validation_output = output[validation]
            validation_loss = F.nll_loss(validation_output, labels[validation]).item()
            validation_accuracy = (validation_output.argmax(dim=1) == labels[validation]).double().mean().item()
            train_accuracy = (output[train_nodes].argmax(dim=1) == train_labels[:, 0]).double().mean().item()
        records.append((epoch, train_loss.item(), train_accuracy, validation_loss, validation_accuracy))
        if epoch in (1, epochs) or epoch % 100 == 0:
            print(f'Epoch {epoch}: train accuracy={train_accuracy:.2%}, '
                  f'validation loss={validation_loss:.6f}, validation accuracy={validation_accuracy:.2%}', flush=True)
    return pd.DataFrame(records, columns=['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'])


def main():
    parser = argparse.ArgumentParser(description='Train a GCN to classify nodes of a synthetic graph.')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.manual_seed(13)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    features, train_labels, validation_labels, adjacency_matrix, G = create_graph()
    features = features.double()
    adjacency = normalize_adjacency(adjacency_matrix.double())
    model = GCN().double()
    history = train(model, features, train_labels, validation_labels, adjacency, args.epochs)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(args.output, index=False)
    if args.plot:
        plot_example(model, features, train_labels, validation_labels, adjacency, G, history)


def plot_example(model, features, train_labels, validation_labels, adjacency, G, history):
    import plotly.express as px
    colors = ['gray', 'purple']
    color_map_train = ['red' for _ in range(200)]
    color_map = [colors[i] for i in validation_labels]

    for i in train_labels:
        color_map_train[i[1]] = colors[i[0]]

    draw_graph(G)

    draw_graph(G, color_map_train)


    draw_graph(G, color_map)

    positions = {i: (features[i, 1], features[i, 2]) for i in range(len(features))}

    draw_graph(G, color_map, positions, axis_labels=('love', 'wealth'))

    positions = {i: (features[i, 0], features[i, 5]) for i in range(len(features))}

    draw_graph(
        G, color_map, positions, axis_labels=('number of dogs', 'age of customer'))

    positions = {i: (features[i, 3], features[i, 4]) for i in range(len(features))}

    draw_graph(G, color_map, positions, axis_labels=(
        'city size', 'healthy eating'))

    with torch.no_grad():
        probability = model(features, adjacency).exp()
        certainty, predicted = probability.max(dim=1)
    positions = {i: (float(predicted[i]), float(certainty[i])) for i in range(len(features))}
    draw_graph(G, color_map, positions, axis_labels=('junk, premium', 'certainty'))
    plt.show()
    long_history = history.melt(id_vars='epoch', var_name='metric', value_name='value')
    px.line(long_history, x='epoch', y='value', facet_row='metric').show()


if __name__ == '__main__':
    main()
