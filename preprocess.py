import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_dist(d):
        
    return (d - d.min()) / (d.max() - d.min() + 1e-8)

def visualize_graph(adj_matrix, threshold=0.25):
    """
    可视化邻接矩阵及其连接特性
    """
    plt.figure(figsize=(15, 5))

    # 邻接矩阵热力图
    plt.subplot(1, 3, 1)
    sns.heatmap(adj_matrix, cmap='viridis', cbar=True)
    plt.title("Fused Adjacency Matrix Heatmap")
    plt.xlabel("Station Index")
    plt.ylabel("Station Index")

    # 稀疏性检查
    plt.subplot(1, 3, 2)
    binary_adj = (adj_matrix > threshold).astype(float)
    plt.imshow(binary_adj, cmap='Greys', interpolation='none')
    plt.title(f"Connectivity (Threshold > {threshold})")

    # 度分布直方图
    plt.subplot(1, 3, 3)
    degrees = np.sum(binary_adj, axis=1) - 1 # 减去自环
    plt.hist(degrees, bins=20, color='skyblue', edgecolor='black')
    plt.title("Node Degree Distribution")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    sparsity = np.mean(binary_adj)
    print(f"--- 图结构统计 ---")
    print(f"矩阵稀疏度: {sparsity:.2%}")
    print(f"平均邻居数: {np.mean(degrees):.2f}")
    
    if sparsity < 0.01:
        print("矩阵过稀疏")
    elif sparsity > 0.5:
        print("矩阵过稠密")

class DSformerPreprocessor:
    def __init__(self, seq_len=144, pred_len=36, sigma=0.5, corr_threshold=0.3):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.sigma = sigma
        self.corr_threshold = corr_threshold
        self.scaler = StandardScaler()
        
    def load_station_data(self, data_dir):
        """
        读取所有CSV,提取经纬度和物理均值用于构建图,同时提取风速序列用于时间相关性计算
        """
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        station_info = []
        all_series = []
        
        print(f"正在读取 {len(files)} 个站点的数据...")
        for f in files:
            path = os.path.join(data_dir, f)
            print(f"正在处理文件: {f}")
            # 提取前两行的经纬度
            meta = pd.read_csv(path, nrows=2, header=None)
            lat, lon = float(meta.iloc[0, 1]), float(meta.iloc[1, 1])
            
            df = pd.read_csv(path, skiprows=2)
            
            # 计算该站点的平均特征（用于空间矩阵 A_s）
            avg_temp = df['Temperature'].mean()
            avg_press = df['Pressure'].mean()
            # 风向转弧度后取平均向量方向
            wd_rad = np.radians(df['Wind Direction'])
            avg_wd = np.arctan2(np.sin(wd_rad).mean(), np.cos(wd_rad).mean())
            
            station_info.append({
                'id': f, 'lat': lat, 'lon': lon, 
                'temp': avg_temp, 'press': avg_press, 'wd': avg_wd
            })
            all_series.append(df['Wind Speed'].values)
            
        return pd.DataFrame(station_info), np.array(all_series)

    def construct_fused_graph(self, station_df, all_speeds):

        n = len(station_df)
        print("正在构建融合邻接矩阵 ...")

        coords = station_df[['lat', 'lon']].values
        d_dist = squareform(pdist(coords)) 
        d_temp = squareform(pdist(station_df[['temp']].values))
        d_press = squareform(pdist(station_df[['press']].values))
        d_wd = squareform(pdist(station_df[['wd']].values))
        d_total = normalize_dist(d_dist) + normalize_dist(d_temp) + normalize_dist(d_press) + normalize_dist(d_wd)
        A_s = np.exp(-(d_total**2) / (2 * self.sigma**2))

        A_t = np.corrcoef(all_speeds)
        A_t[A_t < self.corr_threshold] = 0 

        k = min(n, 10)  
        combined = (A_s + A_t) / 2

        # 使用 NMF 初始化公共基矩阵 W
        nmf_model = NMF(n_components=k, init='nndsvd', random_state=42, max_iter=1000)
        W = nmf_model.fit_transform(combined) 
        
        H_s = np.linalg.lstsq(W, A_s, rcond=None)[0]
        H_t = np.linalg.lstsq(W, A_t, rcond=None)[0]

        H_fused = np.sqrt(np.maximum(H_s * H_t, 0)) 
        
        A_fused = np.dot(W, H_fused)
        # 加入自连接
        A_fused += np.eye(n)

        A_fused = (A_fused - A_fused.min()) / (A_fused.max() - A_fused.min() + 1e-8)
        
        return A_fused

    def process_features(self, data_dir, save_path):

        station_df, all_speeds = self.load_station_data(data_dir)
        A_fused = self.construct_fused_graph(station_df, all_speeds)
        
        # 保存矩阵供模型使用
        np.save(os.path.join(save_path, "adj_matrix.npy"), A_fused)
        print(f"融合邻接矩阵已保存至 {save_path}, 形状: {A_fused.shape}")
        
        return A_fused, station_df
    


if __name__ == "__main__":
    """进行预处理，生成邻接矩阵和站点元数据"""

    processor = DSformerPreprocessor()
    adj, info = processor.process_features("./data", "./output")
    visualize_graph(adj)