import torch
from torch import nn

from model.GCN_layers import GraphConvolution


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.GCN_week = GraphConvolution(time_lag, time_lag)
        self.GCN_day = GraphConvolution(time_lag, time_lag)
        self.GCN_time = GraphConvolution(time_lag, time_lag)
        self.Conv2D = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        nn.init.xavier_normal_(self.Conv2D.weight)
        nn.init.zeros_(self.Conv2D.bias)
        self.linear1 = nn.Linear(8 * time_lag * 3 * station_num, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, station_num * pre_len)
        self.to(device)

    def _encode_history(self, flow, adj):
        week, day, recent = flow.split(self.time_lag, dim=-1)
        return torch.cat((
            self.GCN_week(week, adj),
            self.GCN_day(day, adj),
            self.GCN_time(recent, adj),
        ), dim=-1)

    def forward(self, inflow, outflow, adj):
        inflow_features = self._encode_history(inflow, adj)
        # Mask the second encoded channel while keeping the same model and initialization.
        outflow_features = (
            torch.zeros_like(inflow_features) if outflow is None else self._encode_history(outflow, adj)
        )
        inputs = torch.stack((inflow_features, outflow_features), dim=1)
        features = self.Conv2D(inputs).flatten(start_dim=1)
        features = torch.relu(self.linear1(features))
        features = torch.relu(self.linear2(features))
        return self.linear3(features).reshape(-1, self.station_num, self.pre_len)
