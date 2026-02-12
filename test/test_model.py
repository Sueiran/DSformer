# test_model.py
import torch
from models.dsformer import DSformer

def test_dsformer():
    # 参数设置
    batch_size = 4
    num_nodes = 15
    seq_len = 24  # 测试用短序列
    pred_len = 12
    in_channels = 7 # 我们在 dataset.py 中定义的特征数
    
    # 模拟输入
    x = torch.randn(batch_size, num_nodes, seq_len, in_channels)
    adj = torch.randn(num_nodes, num_nodes) # 模拟 A_fused
    
    # 初始化模型
    model = DSformer(num_nodes, in_channels, seq_len, pred_len)
    
    # 前向传播
    output = model(x, adj)
    
    print(f"模型输入形状: {x.shape}")
    print(f"模型输出形状: {output.shape}")
    
    assert output.shape == (batch_size, num_nodes, pred_len), "输出形状错误！"
    print("DSformer 整体架构搭建成功！")

if __name__ == "__main__":
    test_dsformer()