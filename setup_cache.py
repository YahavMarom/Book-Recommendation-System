import pickle, json
from recommendations import load_data, create_indices
from recommendations import create_description_matrix, create_genres_matrix, create_rating_matrix
from recommendations import stacking_matrix, computing_distances, recommend_books


df = load_data()
indices = create_indices(df)
desc_matrix = create_description_matrix(df)
genre_matrix = create_genres_matrix(df)
rating_matrix = create_rating_matrix(df)


pickle.dump(df, open("data/matrices/dataframe.pkl", "wb"))
pickle.dump(indices, open("data/matrices/indices.pkl", "wb"))
pickle.dump(desc_matrix, open("data/matrices/desc_matrix.pkl", "wb"))
pickle.dump(genre_matrix, open("data/matrices/genre_matrix.pkl", "wb"))
pickle.dump(rating_matrix, open("data/matrices/rating_matrix.pkl", "wb"))

X = stacking_matrix(desc_matrix, genre_matrix, rating_matrix)
knn = computing_distances(X)

book_titles = {}
cached = {}

for title in df["Book"]:
    book_titles[title] = title
    recs = recommend_books(df, X, knn, indices, title, 5)
    cached[title] = recs

json.dump(cached, open("data/cached_recommendations.json", "w"))
json.dump(book_titles, open("data/book_titles.json", "w"))
