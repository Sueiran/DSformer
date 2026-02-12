# test_layers.py
import torch
from layers.drgcn import DeepResGCNLayer, IntegratedWindowAttention

def test_layers():
    # 模拟数据
    batch_size = 8
    nodes = 15
    seq_len = 24  # 你刚才改成功的参数
    d_model = 64
    
    # 测试 DeepResGCN
    x = torch.randn(batch_size, nodes, d_model)
    adj = torch.randn(nodes, nodes)
    gcn = DeepResGCNLayer(d_model, d_model)
    out_gcn = gcn(x, adj, x)
    print(f"GCN 输出形状: {out_gcn.shape}") # 应为 [8, 15, 64]
    
    # 测试 IWA
    # 输入通常是 [Batch*Nodes, Seq_Len, d_model]
    x_ts = torch.randn(batch_size * nodes, seq_len, d_model)
    iwa = IntegratedWindowAttention(d_model, num_heads=4, window_size=6)
    out_iwa = iwa(x_ts)
    print(f"IWA 输出形状: {out_iwa.shape}") # 应为 [120, 24, 64]

if __name__ == "__main__":
    test_layers()