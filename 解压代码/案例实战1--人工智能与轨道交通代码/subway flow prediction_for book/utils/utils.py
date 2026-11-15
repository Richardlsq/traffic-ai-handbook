import numpy as np
from scipy import sparse


class GetLaplacian:
    def __init__(self, adjacency):
        self.adjacency = np.asarray(adjacency)

    def get_normalized_adj(self, station_num):
        if self.adjacency.shape != (station_num, station_num):
            raise ValueError(f'Expected a {station_num} by {station_num} adjacency matrix')
        return self._normalized_with_self_loops().toarray().astype(np.float32)

    def normalized_adj(self, adjacency=None):
        graph = sparse.coo_matrix(self.adjacency if adjacency is None else adjacency)
        degree = np.asarray(graph.sum(axis=0)).ravel()
        inverse_roots = np.zeros(degree.shape, dtype=np.float64)
        nonzero = degree > 0
        inverse_roots[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        scale = sparse.diags(inverse_roots)
        return (scale @ graph @ scale).tocoo()

    def _normalized_with_self_loops(self):
        return self.normalized_adj(self.adjacency + np.eye(self.adjacency.shape[0]))

    def sparse_to_tuple(self, matrix):
        matrix = matrix.tocoo()
        return np.column_stack((matrix.row, matrix.col)), matrix.data, matrix.shape

    def calculate_laplacian(self):
        return self.sparse_to_tuple(self._normalized_with_self_loops())
