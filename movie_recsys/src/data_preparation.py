import pandas as pd

def load_movielens_data(path='data/ml-1m/'):
    # ratings.dat 格式：UserID::MovieID::Rating::Timestamp
    ratings = pd.read_csv(path + 'ratings.dat', 
                          sep='::', 
                          names=['user_id', 'movie_id', 'rating', 'timestamp'],
                          engine='python')
    # movies.dat 格式：MovieID::Title::Genres
    movies = pd.read_csv(path + 'movies.dat',
                         sep='::',
                         names=['movie_id', 'title', 'genres'],
                         engine='python')
    
    data = pd.merge(ratings, movies, on='movie_id')
    # 可以再根据需要进行数据清洗，如去重、timestamp处理等
    return data