# GADF Stock XAI

美股 GADF-CNN-LRP 可解释性预测流程

## 项目目标

在 Kaggle Notebook 中一条龙运行美股 GADF-CNN-LRP 可解释性预测流程。

## 核心流程

```
美股数据 → 特征工程 → 滑动窗口 → GADF 图像编码 → CNN 预测未来涨跌 → LRP 解释热力图 → Spectral Clustering 聚类 → 输出预测结果、热力图和聚类结果
```

## 项目结构

```
gadf_stock_xai/
├── configs/                 # 配置文件
│   └── spy_daily.yaml      # SPY 日线配置
├── src/                    # 源代码
│   ├── data_downloader.py  # 数据下载
│   ├── indicators.py       # 技术指标
│   ├── make_dataset.py     # 数据集构建
│   ├── gadf_encoder.py     # GADF 编码
│   ├── models.py           # CNN 模型
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── lrp_explainer.py    # LRP 解释器
│   ├── spray_cluster.py    # 谱聚类
│   ├── visualization.py    # 可视化
│   └── utils.py            # 工具函数
├── scripts/                # 运行脚本
│   └── run_all.py          # 主流程脚本
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后数据
├── outputs/               # 输出目录
│   ├── reports/           # 报告
│   ├── models/            # 模型
│   ├── heatmaps/          # 热力图
│   ├── relevance_maps/    # 相关性图
│   └── clusters/          # 聚类结果
├── requirements.txt       # 依赖
└── README.md             # 说明文档
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整流程

```bash
python scripts/run_all.py --config configs/spy_daily.yaml
```

## 流程步骤

1. **Step 1**: Download data - 下载美股数据
2. **Step 2**: Build features and labels - 构建特征和标签
3. **Step 3**: Create sliding windows - 创建滑动窗口
4. **Step 4**: Encode GADF images - GADF 图像编码
5. **Step 5**: Train SimpleCNN - 训练 CNN 模型
6. **Step 6**: Evaluate on test set - 测试集评估
7. **Step 7**: Generate LRP heatmaps - 生成 LRP 热力图
8. **Step 8**: Run spectral clustering - 运行谱聚类
9. **Step 9**: Save final outputs - 保存最终输出

## 配置说明

主要配置项在 `configs/spy_daily.yaml` 中：

- **data**: 数据源配置（股票代码、日期范围等）
- **features**: 特征工程配置（RSI、BIAS 等指标参数）
- **label**: 标签生成配置（预测窗口、阈值等）
- **dataset**: 数据集配置（窗口大小、图像大小）
- **model**: 模型配置（CNN 结构、训练参数）
- **lrp**: LRP 解释配置
- **clustering**: 聚类配置

## 依赖库

- yfinance: 股票数据下载
- numpy, pandas: 数据处理
- scikit-learn: 机器学习工具
- matplotlib: 可视化
- pyts: 时间序列图像编码
- torch, torchvision: 深度学习
- captum: 可解释性 AI
- pyyaml: 配置管理
- tqdm: 进度条

## 开发状态

🚧 第一版：最小可运行闭环（进行中）

优先保证流程能跑通，模型准确率不是第一优先级。

## License

MIT
