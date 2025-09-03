from flask import Flask, render_template, request
from recommendations import load_data, recommend_books, stacking_matrix, computing_distances
import pickle, json



df = pickle.load(open("data/matrices/dataframe.pkl", "rb"))
indices = pickle.load(open("data/matrices/indices.pkl", "rb"))
desc_matrix = pickle.load(open("data/matrices/desc_matrix.pkl", "rb"))
genre_matrix = pickle.load(open("data/matrices/genre_matrix.pkl", "rb"))
rating_matrix = pickle.load(open("data/matrices/rating_matrix.pkl", "rb"))
cached_recommendations = json.load(open("data/cached_recommendations.json"))


app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    book_title = request.form["book_title"].lower().strip()
    if book_title in cached_recommendations:
        recommendations = cached_recommendations[book_title]
        return render_template("results.html", recommendations=recommendations, book=book_title)
    return render_template("error.html", message= f"No recommendations found for {book_title}")


@app.route("/custom")
def custom():
    book = request.args.get("book")
    if not book:
        return "Incorrect book", 400
    weights = [i/10 for i in range(1, 9)]
    return render_template("custom.html", weights = weights, book=book)


@app.route("/search_custom", methods=["POST"])
def search_custom():
    book = request.form.get("book")
    if not book:
        return "incorrect book", 400
    try:
        w1 = float(request.form["w1"])
        w2 = float(request.form["w2"])
    except (TypeError, ValueError):
        return render_template("custom.html", book=book, weights = [i/10 for i in range(1, 9)],  message = "Invalid input")
    
    w1 = round(w1, 1)
    w2 = round(w2, 1)
    w3 = round(1-w1-w2, 1)

    if ( w1+w2 >= 1 ) or (w3 >= 1) or (w3 < 0.1) or not (0.1 <= w1 <= 0.9) or not (0.1 <= w2 <= 0.9):
        print(f"w1 = {w1}, w2 = {w2}, w3 = {w3}")
        return render_template("custom.html", book=book, weights = [i/10 for i in range(1, 9)], message="each weight must be in 0.1-0.9, and w1+w2 must be < 1")
    
    Y = stacking_matrix(desc_matrix, genre_matrix, rating_matrix, w1, w2, w3)
    knn = computing_distances(Y)
    recommendations = recommend_books(df, Y, knn, indices, book)

    return render_template("results.html", recommendations=recommendations, book=book)

if __name__ == "__main__":
    app.run(debug=True)
