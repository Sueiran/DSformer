# test_preprocess.py
import os
import numpy as np
import pandas as pd
from preprocess import DSformerPreprocessor

def create_dummy_data(data_dir, num_stations=15):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for i in range(num_stations):
        # 模拟你的文件格式
        lat, lon = 30.0 + i*0.1, 120.0 + i*0.1
        data = {
            'Hour': np.tile(np.arange(24), 10),
            'Month': np.ones(240),
            'Temperature': np.random.normal(20, 5, 240),
            'Pressure': np.random.normal(1010, 10, 240),
            'Wind Direction': np.random.randint(0, 360, 240),
            'Wind Speed': np.random.normal(5, 2, 240)
        }
        df = pd.DataFrame(data)
        with open(os.path.join(data_dir, f"station_{i+1}.csv"), 'w') as f:
            f.write(f"lat,{lat}\n")
            f.write(f"lon,{lon}\n")
            df.to_csv(f, index=False)

if __name__ == "__main__":
    # 1. 生成测试数据
    create_dummy_data("./test_csvs")
    
    # 2. 运行预处理
    processor = DSformerPreprocessor()
    if not os.path.exists("./output"): os.makedirs("./output")
    adj, info = processor.process_features("./test_csvs", "./output")
    
    # 3. 验证
    print("--- 测试结果 ---")
    assert adj.shape == (15, 15), "矩阵维度错误！"
    assert not np.isnan(adj).any(), "矩阵包含空值！"
    print("邻接矩阵检查通过！")
    print("站点元数据提取结果：")
    print(info.head())