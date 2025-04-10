# src/baseline_logistic.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    return lr

def evaluate_logistic_regression(model, X_val, y_val):
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_prob)
    print(f"Logistic Regression Validation AUC: {auc:.4f}")
    return auc

if __name__ == '__main__':
    # 示意：生成一些简单特征进行逻辑回归训练
    import pandas as pd
    from feature_engineering import load_movielens_data, build_features, get_train_val_test
    df = load_movielens_data()
    df = build_features(df)
    train, val, _ = get_train_val_test(df)

    # 构造简单特征矩阵：例如用户_click_rate和user_interactions作为特征
    X_train = train[['user_click_rate', 'user_interactions']].values
    y_train = train['label'].values
    X_val = val[['user_click_rate', 'user_interactions']].values
    y_val = val['label'].values

    model = train_logistic_regression(X_train, y_train)
    evaluate_logistic_regression(model, X_val, y_val)