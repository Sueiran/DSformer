import torch
import torch.nn as nn
from layers.drgcn import DeepResGCNLayer, IntegratedWindowAttention

class DSformer(nn.Module):
    def __init__(self, num_nodes, in_channels, seq_len, pred_len, d_model=64, num_layers=3, num_heads=4):
        super(DSformer, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.embedding = nn.Linear(in_channels, d_model)
        
        # DeepResGCN 
        self.gcn_layers = nn.ModuleList([
            DeepResGCNLayer(d_model, d_model) for _ in range(num_layers)
        ])
        
        # SMARTformer 
        self.time_layers = nn.ModuleList([
            IntegratedWindowAttention(d_model, num_heads, window_size=6)
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(d_model * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  ## 防止过拟合
            nn.Linear(256, pred_len)
        )

    def forward(self, x, adj):
        B, N, L, F = x.shape
        
        x = self.embedding(x)
        x_gcn = x.transpose(1, 2).reshape(B * L, N, -1)
        h0 = x_gcn 
        
        for layer in self.gcn_layers:
            x_gcn = layer(x_gcn, adj, h0)

        x = x_gcn.view(B, L, N, -1).transpose(1, 2)
        
        x_time = x.reshape(B * N, L, -1)
        for layer in self.time_layers:
            x_time = layer(x_time)

        x_out = x_time.reshape(B * N, -1)
        ## 只取最后一个时间步的特征进行预测
        # x_out = x_time[:, -1, :]
        ## 取时间维度的平均值
        # x_out = torch.mean(x_time, dim=1) 
        forecast = self.projection(x_out)
        return forecast.view(B, N, self.pred_len)