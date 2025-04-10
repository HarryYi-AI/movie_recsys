# # src/evaluation.py
# import numpy as np
# from sklearn.metrics import roc_auc_score

# def compute_auc(y_true, y_pred):
#     return roc_auc_score(y_true, y_pred)

# def precision_at_k(y_true, y_pred, k=10):
#     """
#     计算precision@k
#     y_pred为预测概率或排序得分，y_true为真实标签，假设按照得分降序排序取前k预测正例，比较其中的真实正例比例
#     """
#     idx_sorted = np.argsort(y_pred)[::-1]
#     top_k_idx = idx_sorted[:k]
#     precision = np.sum(y_true[top_k_idx]) / k
#     return precision

# if __name__ == '__main__':
#     # 示例调用
#     y_true = np.array([1, 0, 1, 0, 1])
#     y_pred = np.array([0.9, 0.2, 0.8, 0.3, 0.7])
#     print("AUC:", compute_auc(y_true, y_pred))
#     print("Precision@3:", precision_at_k(y_true, y_pred, k=3))

# src/evaluation.py
import numpy as np
from sklearn.metrics import roc_auc_score
from data_preparation import load_movielens_data
from feature_engineering import build_features, get_train_val_test, prepare_training_data
import keras
from keras import load_model

def compute_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def precision_at_k(y_true, y_pred, k=10):
    """
    计算 precision@k
    y_pred 为预测概率或排序得分，y_true 为真实标签，假设按照得分降序排序取前 k 个样本中真实为正例的比例
    """
    idx_sorted = np.argsort(y_pred)[::-1]
    top_k_idx = idx_sorted[:k]
    precision = np.sum(y_true[top_k_idx]) / k
    return precision

if __name__ == '__main__':
    # 1. 数据加载与特征构造（使用 MovieLens 数据）
    df = load_movielens_data()
    df = build_features(df)
    train_df, val_df, test_df = get_train_val_test(df)
    
    # 2. 准备测试数据：生成 DIN 模型所需的输入数据，假设历史行为序列长度为 10
    seq_len = 10
    test_data, y_test, _, _ = prepare_training_data(test_df, seq_len=seq_len)
    
    # 3. 加载训练好的模型（模型训练完成后，你一般会保存模型文件，例如保存为 saved_model/din_model.h5）
    # 请确保模型已经保存
    model = load_model('saved_model/din_model.h5')
    
    # 4. 对测试数据进行预测
    # 注意：此处使用字典传入各个输入项，确保这些名称与模型构建时的 Input 名称一致
    y_pred = model.predict({
        'user_input': test_data[0],
        'item_input': test_data[1],
        'hist_input': test_data[2]
    })
    
    # 5. 计算评价指标
    auc = compute_auc(y_test, y_pred)
    # 由于 y_pred 返回为 (samples, 1) 数组，展平后传递给 precision_at_k
    precision = precision_at_k(y_test, y_pred.flatten(), k=3)
    
    print("Test AUC:", auc)
    print("Test Precision@3:", precision)