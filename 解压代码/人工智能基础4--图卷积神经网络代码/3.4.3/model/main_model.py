import torch
from torch import nn
import torch.nn.functional as F
from model.GCN_layers import GraphConvolution

class Model(nn.Module):
	def __init__(self, time_lag, pre_len, station_num, device):
		super().__init__()
		self.time_lag = time_lag
		self.pre_len = pre_len
		self.station_num = station_num
		self.device = device
		self.GCN1 = GraphConvolution(in_features=self.time_lag , out_features=self.time_lag).to(self.device)
		self.GCN2 = GraphConvolution(in_features=self.time_lag , out_features= self.time_lag).to(self.device)
		self.linear1 = nn.Linear(in_features= self.time_lag * self.station_num, out_features=1024).to(self.device)
		self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
		self.linear3 = nn.Linear(in_features=512, out_features=self.station_num * self.pre_len).to(self.device)
	def forward(self, speed, adj):
		speed = speed.to(self.device)  #[32,276,10]
		adj = adj.to(self.device)
		speed = self.GCN1(x=speed, adj = adj)  # (32, 276, 10)
		output = self.GCN2(x = speed, adj = adj)  #[32, 276, 10]
		output = output.reshape(output.size()[0], -1)  # (32, 276*10)
		output = F.relu(self.linear1(output))  # (32, 1024)
		output = F.relu(self.linear2(output))  # (32, 512)
		output = self.linear3(output)  # (32, 276*pre_len)
		output = output.reshape(output.size()[0], self.station_num, self.pre_len)  # ( 64, 276, pre_len)
		return output
