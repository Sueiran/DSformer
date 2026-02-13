# DSformer: Integration of DeepResGCN and SMARTformer for Enhanced Spatiotemporal Wind Speed Forecasting

## Overview

This repository contains the paper "DSformer: Integration of DeepResGCN and SMARTformer for Enhanced Spatiotemporal Wind Speed Forecasting" by Chenxing Zhu, Yizhi Dong, Zijun Ye, and Wenbo Wang. The DSformer model is a novel approach for short-term wind speed forecasting across multiple wind farms, combining Deep Residual Graph Convolutional Networks (DeepResGCN) for spatial feature extraction and Semi-Autoregressive Transformer (SMARTformer) for temporal dependency modeling.

The model addresses the limitations of existing methods by fully exploiting spatiotemporal correlations among neighboring wind farms, using dynamic graph construction, multi-graph feature fusion via low-rank decomposition, and integrated window attention mechanisms.

Key highlights:
- Achieves superior performance in 4-, 6-, and 8-hour forecasts on a dataset from 15 wind farms.
- MAE: 0.273 m/s (4h), 0.291 m/s (6h), 0.357 m/s (8h)
- RMSE: 0.147 m/s (4h), 0.189 m/s (6h), 0.238 m/s (8h)
- Outperforms baselines like PatchTST, Informer, LSTM, SMARTformer, and LightGBM.

The paper is available as [DSformer.pdf](./DSformer.pdf).

## Authors

- Chenxing Zhu (Undergraduate, Statistics, Wuhan University of Science and Technology) - 13667121392@163.com
- Yizhi Dong (Undergraduate, Statistics, Wuhan University of Science and Technology) - sueiran42526@163.com
- Zijun Ye (Undergraduate, Statistics, Wuhan University of Science and Technology) - yezijunchongyaaa@163.com
- Wenbo Wang (Professor, Wuhan University of Science and Technology) - wangwenbo@wust.edu.cn (Corresponding Author)

## Methodology

### Spatial Feature Extraction (DeepResGCN)
- **Graph Construction**: Builds spatial and temporal adjacency matrices using geographical distances, wind directions, temperatures, pressures, and time-lagged correlations.
- **Adjacency Matrix Fusion**: Uses joint low-rank decomposition (NMF) and Hadamard product for fusing static and dynamic features.
- **Multi-layer Stacked GCN**: Employs residual connections and weighted pooling to capture higher-order neighborhood information.

### Temporal Feature Extraction (SMARTformer)
- **Time-Independent Embedding**: Combines value embedding with hierarchical positional embeddings for minutes/hours, weekdays, and months.
- **Integrated Window Attention**: Splits attention heads into intra-window and inter-window branches for efficient local and global dependency capture.
- **Semi-Autoregressive Decoder**: Hierarchical design with segment AR layers for local details and NAR refining layers for global horizons.

The overall DSformer framework integrates these components for robust spatiotemporal forecasting.

## Dataset
- Source: Inner Mongolia region from the Asia-Pacific Sunflower Solar Dataset (WIND Toolkit).
- Features: Wind speed, direction, temperature, pressure from 15 wind farms in 2020.
- Resolution: 10-minute intervals, spatial distances 7-70 km.
- Split: 70% training, 10% validation, 20% test.

## License
This paper is provided for academic and research purposes. All rights reserved by the authors.
