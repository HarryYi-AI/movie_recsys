import pandas as pd
import numpy as np

def build_features(data: pd.DataFrame):
    # 1. 用户侧特征
    user_group = data.groupby('user_id').agg({
        'rating': ['mean', 'count']
    })
    user_group.columns = ['user_avg_rating', 'user_rating_count']
    user_group.reset_index(inplace=True)
    
    # 2. 电影侧特征
    # 可拆分genres做OneHot，也可建立Embedding映射
    data['genre_list'] = data['genres'].apply(lambda x: x.split('|'))
    
    # join回去，得到 data + user侧特征
    data = pd.merge(data, user_group, on='user_id', how='left')
    
    # 3. 返回数据
    return data