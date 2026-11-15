import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Metrics:
    def __init__(self, Y_true, Y_pred):
        self.Y_true = np.asarray(Y_true, dtype=np.float64)
        self.Y_pred = np.asarray(Y_pred, dtype=np.float64)

    def weighted_mean_absolute_percentage_error(self):
        denominator = np.abs(self.Y_true).sum()
        if denominator == 0:
            return float('nan')
        return float(np.abs(self.Y_true - self.Y_pred).sum() / denominator)

    def evaluate_performance(self):
        rmse = float(np.sqrt(mean_squared_error(self.Y_true, self.Y_pred)))
        r2 = float(r2_score(self.Y_true, self.Y_pred)) if len(self.Y_true) > 1 else float('nan')
        mae = float(mean_absolute_error(self.Y_true, self.Y_pred))
        return rmse, r2, mae, self.weighted_mean_absolute_percentage_error()


class Metrics_1d(Metrics):
    pass
