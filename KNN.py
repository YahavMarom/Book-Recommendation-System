import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.sparse import hstack, csr_matrix
import re



def remove_parens(x):
    return re.sub(r"[\(\[].*?[\)\]]", "", x).strip()



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
    df['Book'] = df['Book'].apply(remove_parens)

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
    scalar = MinMaxScaler()
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
    knn = NearestNeighbors(metric='cosine')
    knn.fit(X)

    return knn


def create_indices(df):
    return df.Series(df.index, index=df['Book'])


# n books, all from different authors.
def recommend_books(df, X, knn, indices, title, top_n = 5):
    d = set()
    if title not in indices:
        print("Book not in database. Make sure book is spelled correctly.")
    else:
        index = indices[title]
        distances, neighbor_id = knn.kneighbors(X[index], top_n+20)

        result = []
        for dist, id in zip(distances[0][1:], neighbor_id[0][1:]):
            row = df.iloc[id]
            if d is None or row['Author'] not in d:

                similarity_score = round(1 - dist, 3)
                result.append(
                    {
                        "title": row['Book'],
                        "rating" : float(row['Avg_Rating']),
                        "similarity score": float(similarity_score),
                        "author" : row['Author']
                    }

                )
            d.add(row['Author'])
            if len(result) >= top_n:
               break

            
        return result

def print_list(lst):
    for line in lst:
        print(line)


def full_algorithm():
    df = load_data()
    desc_matrix = create_description_matrix(df)
    genre_matrix = create_genres_matrix(df)
    rating_matrix = create_rating_matrix(df)
    X = stacking_matrix(desc_matrix, genre_matrix, rating_matrix)

    knn = computing_distances(X)
    indices = create_indices(df)
    title_to_test = "To Kill a Mockingbird"

    print_list(recommend_books(df, X, knn, indices, title_to_test, top_n=5))

def print_list(lst):
    for line in lst:
        print(line)

if __name__ == "__main__":
    full_algorithm()
    

