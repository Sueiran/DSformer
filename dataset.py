import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

class DSformerDataset(Dataset):
    def __init__(self, data_dir, seq_len=24, pred_len=12, mode='train'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        
        self.data, self.mean, self.std = self._read_and_align_data(data_dir)
        
        num_samples = self.data.shape[0]
        train_end = int(num_samples * 0.7)
        val_end = int(num_samples * 0.8)
        
        if mode == 'train':
            self.data = self.data[:train_end]
        elif mode == 'val':
            self.data = self.data[train_end:val_end]
        else:
            self.data = self.data[val_end:]
            
        self.valid_indices = self._get_indices()

    def _read_and_align_data(self, data_dir):
        """
        核心：将 15 个 CSV 文件转化为 [Total_Time, Nodes, Features] 的 3D 张量
        """
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        all_station_data = []
        
        for f in files:
            df = pd.read_csv(os.path.join(data_dir, f), skiprows=2)
            # 特征选择：风速(target), 温度, 气压, WD_sin, WD_cos, Hour_sin, Hour_cos
            df['WD_sin'] = np.sin(np.radians(df['Wind Direction']))
            df['WD_cos'] = np.cos(np.radians(df['Wind Direction']))
            df['H_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['H_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            
            feats = df[['Wind Speed', 'Temperature', 'Pressure', 'WD_sin', 'WD_cos', 'H_sin', 'H_cos']].values
            all_station_data.append(feats)
        
        combined_data = np.stack(all_station_data, axis=1).astype(np.float32)
        
        mean = combined_data.mean(axis=(0, 1), keepdims=True)
        std = combined_data.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1.0
        
        normalized_data = (combined_data - mean) / std
        return normalized_data, mean, std

    def _get_indices(self):
        return np.arange(len(self.data) - self.seq_len - self.pred_len + 1)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len
        target_end = end + self.pred_len
        
        x = self.data[start:end].transpose(1, 0, 2)
        
        y = self.data[end:target_end, :, 0].transpose(1, 0)
        
        return torch.from_numpy(x), torch.from_numpy(y)