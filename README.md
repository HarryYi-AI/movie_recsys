# MovieLens Recommendation System with DIN

## 项目简介
本项目基于MovieLens数据集构建一个智能推荐系统，主要实现了两种模型：
- **Baseline：** 使用逻辑回归模型，对用户点击率等简单特征进行预测。
- **DIN模型：** 利用Deep Interest Network，将用户历史行为序列与目标物品Embedding结合进行点击率预测。

## 数据准备
- 数据集：MovieLens 1M (可从 [MovieLens官网](https://grouplens.org/datasets/movielens/1m/) 下载)
- 数据预处理及特征构造在 `src/data_preparation.py` 和 `src/feature_engineering.py` 中实现。

## 项目结构
movie_recsys
├── data
│   └── ml-1m      # 放MovieLens数据集，如 ratings.dat、movies.dat 等
├── src
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── baseline_logistic.py
│   ├── model_din.py
│   ├── train.py
│   ├── evaluation.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore

## 环境与依赖
- Python 3.8+
- 主要Python库：pandas, numpy, scikit-learn, tensorflow, keras

## 具体操作
数据预处理：  
 ```bash
 python src/data_preparation.py
 ```

特征构造与训练：
bash
python src/train.py

查看模型评估结果与指标计算：
bash
python src/evaluation.py
