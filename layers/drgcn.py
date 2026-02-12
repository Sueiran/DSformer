import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepResGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=0.1, theta=0.5, dropout=0.2):
        super(DeepResGCNLayer, self).__init__()
        self.alpha = alpha
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bn = nn.BatchNorm1d(out_channels) # BN 稳定分布
        self.dropout = nn.Dropout(dropout)      ## 防止过拟合
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj, h0):

        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        
        output = (1 - self.alpha) * output + self.alpha * h0
        
        BL, N, D = output.shape
        output = output.view(-1, D)
        output = self.bn(output)
        output = output.view(BL, N, D)
        
        return F.relu(self.dropout(output))

class IntegratedWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super(IntegratedWindowAttention, self).__init__()
        self.window_size = window_size
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        ###### 层归一化
        self.norm = nn.LayerNorm(d_model) 
        
    def forward(self, x):

        residual = x 
        B_N, L, D = x.shape
        w = self.window_size
        
        num_windows = L // w
        x_win = x.view(B_N, num_windows, w, D).reshape(-1, w, D)
        
        attn_out, _ = self.mha(x_win, x_win, x_win)
        
        out = attn_out.reshape(B_N, L, D)
        # 加上残差连接和归一化
        return self.norm(out + residual)

class MultiGraphFeatureFusion(nn.Module):

    def __init__(self, feature_dim):
        super(MultiGraphFeatureFusion, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim))
        
    def forward(self, spatial_feat, temporal_feat):

        return spatial_feat * self.weight + temporal_feat * (1 - self.weight)
    

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_metric, model):
        score = -val_metric 
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0