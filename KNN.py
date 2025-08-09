import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.sparse import hstack, csr_matrix

def load_data():
    df = pd.read_csv('goodreads_data.csv')

    df = df.drop(df.columns[[0]], axis=1)

    df['Description'] = df['Description'].fillna('')
    df['Genres'] = df['Genres'].fillna('')

    df['Avg_Rating'] = pd.to_numeric(df['Avg_Rating'], errors='coerce')

    df['Num_Ratings'] = df['Num_Ratings'].replace(',', '', regex=True)
    df['Num_Ratings'] = pd.to_numeric(df['Num_Ratings'], errors='coerce').fillna(0).astype(int)

    df = df[df['Num_Ratings'] >= 10].reset_index(drop=True)
    df = df.drop_duplicates(subset=['Book'], keep='first').reset_index(drop=True)

    return df

# using tf-idf for the description
def create_description_matrix(df):
    vectorizer = TfidfVectorizer(max_df = 0.8, min_df=2, stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df['Description'])
    
    return tfidf_matrix


def create_genres_matrix(df):
    vectorizer = CountVectorizer(token_pattern='[^,|;]+', lowercase=True, strip_accents='ascii')
    genre_matrix = vectorizer.fit_transform(df['Genres'])

    return genre_matrix


def create_rating_matrix(df):
    scalar = MinMaxScaler
    rating_matrix = scalar.fit_transform(df[ ['Avg_Rating', 'Num_Ratings'] ])
    rating_matrix = csr_matrix(rating_matrix)

    return rating_matrix


# normalizing the matrices, giving a weight to each matrix. (D gets alpha, G gets beta, R gets gamma)
def stacking_matrix(D, G, R, alpha=0.5, beta=0.3, gamma = 0.2):
    Dn = normalize(D, 'l2') * np.sqrt(alpha)
    Gn = normalize(G, 'l2') * np.sqrt(beta)
    Rn = normalize(R, 'l2') * np.sqrt(gamma)

    X = hstack( [Dn, Gn, Rn] )
    return X

def computing_distances(X):
    similarity_matrix = cosine_similarity(X, X)
    knn = NearestNeighbors(metric='cosine')
    knn.fit(X)

    return knn


def recommended_books(knn, indices, title, top_n = 3):
    idx = indices[title]


def full_algorithm():

    df = load_data()

    desc_matrix = create_description_matrix(df)
    genre_matrix = create_genres_matrix(df)
    rating_matrix = create_rating_matrix(df)
    X = stacking_matrix(desc_matrix, genre_matrix, rating_matrix)







if __name__ == "__main__":
    df = load_data()
    desc_matrix = create_description_matrix(df)
    genre_matrix = create_genres_matrix(df)
    rating_matrix = create_rating_matrix(df)
    X = stacking_matrix(desc_matrix, genre_matrix, rating_matrix)

    knn = computing_distances(X)
    indices = pd.Series(df.index, index=df['title'])

    