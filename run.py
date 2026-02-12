import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from dataset import DSformerDataset
from models.dsformer import DSformer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from layers.drgcn import EarlyStopping

def train():
    # --- 日志配置 ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("./output/train.log", mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    # --- 1. 超参数设置 ---
    parser = argparse.ArgumentParser(description="DSformer训练参数")
    parser.add_argument('--num_nodes', type=int, default=15, help='节点数')
    parser.add_argument('--in_channels', type=int, default=7, help='输入通道数')
    parser.add_argument('--seq_len', type=int, default=144, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=36, help='预测序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--early_stop', action='store_true', help='是否启用早停')
    parser.add_argument('--patience', type=int, default=5, help='早停容忍轮数')
    parser.add_argument('--min_delta', type=float, default=0.0, help='早停最小改变量')
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("使用 Apple Silicon GPU (MPS) 进行训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("使用 CUDA GPU 进行训练")
    else:
        device = torch.device("cpu")
        logging.info("使用 CPU 进行训练")

    num_nodes = args.num_nodes
    in_channels = args.in_channels
    seq_len = args.seq_len
    pred_len = args.pred_len
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # --- 2. 加载数据与邻接矩阵 ---
    adj = np.load("./output/adj_matrix.npy")
    adj = torch.from_numpy(adj).float().to(device)
    
    train_dataset = DSformerDataset("./data", seq_len, pred_len, mode='train')
    val_dataset = DSformerDataset("./data", seq_len, pred_len, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # --- 3. 初始化模型、损失函数与优化器 ---
    model = DSformer(num_nodes, in_channels, seq_len, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_mean = torch.tensor(train_dataset.mean[0, 0, 0]).to(device)
    data_std = torch.tensor(train_dataset.std[0, 0, 0]).to(device)

    logging.info("开始训练...")
    # 记录指标
    history = {
        'epoch': [],
        'train_loss': [],
        'train_mae': [],
        'train_mse': [],
        'val_mae': [],
        'val_mse': []
    }

    # --- 早停机制初始化 ---
    if args.early_stop:
        early_stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        logging.info(f"启用早停，patience={args.patience}, min_delta={args.min_delta}")
    else:
        early_stopper = None

    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_mae = []
        train_mse = []
        logging.info(f"Epoch {epoch+1}/{epochs} 正在训练...")
        for x, y in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x, adj) 
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # 逆归一化还原为真实风速
            real_pred = output * data_std + data_mean
            real_y = y * data_std + data_mean
            mae = torch.abs(real_pred - real_y).mean()
            mse = ((real_pred - real_y) ** 2).mean()
            train_mae.append(mae.item())
            train_mse.append(mse.item())

        # --- 验证环节 ---
        model.eval()
        val_mae = []
        val_mse = []
        logging.info(f"Epoch {epoch+1}/{epochs} 正在验证...")
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
                x, y = x.to(device), y.to(device)
                pred = model(x, adj)
                # 逆归一化还原为真实风速
                real_pred = pred * data_std + data_mean
                real_y = y * data_std + data_mean

                mae = torch.abs(real_pred - real_y).mean()
                mse = ((real_pred - real_y) ** 2).mean()
                val_mae.append(mae.item())
                val_mse.append(mse.item())
        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(train_loss):.4f} | Train MAE: {np.mean(train_mae):.4f} | Train MSE: {np.mean(train_mse):.4f} | Val MAE: {np.mean(val_mae):.4f} m/s | Val MSE: {np.mean(val_mse):.4f}")

        # --- 早停判断 ---
        if early_stopper is not None:
            early_stopper(np.mean(val_mae), model)
            if early_stopper.early_stop:
                logging.info(f"早停触发，训练在第{epoch+1}轮提前终止。")
                # 恢复最佳模型参数
                model.load_state_dict(early_stopper.best_state)
                break

        # 记录本轮指标
        history['epoch'].append(epoch+1)
        history['train_loss'].append(np.mean(train_loss))
        history['train_mae'].append(np.mean(train_mae))
        history['train_mse'].append(np.mean(train_mse))
        history['val_mae'].append(np.mean(val_mae))
        history['val_mse'].append(np.mean(val_mse))

    # --- 保存训练过程指标为csv ---
    df = pd.DataFrame(history)
    df.to_csv('./output/train_history.csv', index=False)
    logging.info('训练过程指标已保存为 ./output/train_history.csv')

    # --- 可视化loss、MAE和MSE曲线 ---
    plt.figure(figsize=(18,4))
    plt.subplot(1,4,1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,4,2)
    plt.plot(history['epoch'], history['train_mae'], label='Train MAE', color='blue')
    plt.plot(history['epoch'], history['val_mae'], label='Val MAE', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (m/s)')
    plt.title('MAE')
    plt.legend()

    plt.subplot(1,4,3)
    plt.plot(history['epoch'], history['train_mse'], label='Train MSE', color='purple')
    plt.plot(history['epoch'], history['val_mse'], label='Val MSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('MSE')
    plt.legend()

    plt.subplot(1,4,4)
    plt.plot(history['epoch'], history['val_mae'], label='Val MAE', color='orange')
    plt.plot(history['epoch'], history['val_mse'], label='Val MSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Val MAE & MSE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./output/train_history.png')
    plt.close()
    logging.info('训练过程曲线已保存为 ./output/train_history.png')

if __name__ == "__main__":
    train()