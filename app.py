import json
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy import sparse

from recommendations import build_blocks, create_indices, recommend_books

df = pickle.load(open("data/matrices/dataframe.pkl", "rb"))
indices = create_indices(df)
blocks = build_blocks(
    np.load("data/matrices/desc.npy"),
    sparse.load_npz("data/matrices/genre.npz"),
    np.load("data/matrices/rating.npy"),
)
titles = json.load(open("data/book_titles.json"))
try:
    covers = json.load(open("data/covers.json"))
except FileNotFoundError:
    covers = {}

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/titles")
def api_titles():
    resp = jsonify(titles)
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp


@app.route("/api/recommend")
def api_recommend():
    key = request.args.get("book", "").lower().strip()
    if key not in indices:
        return jsonify(error=f"'{key}' is not in the database"), 404

    def num(name, default, lo, hi, cast=float):
        try:
            return min(max(cast(request.args.get(name, default)), lo), hi)
        except (TypeError, ValueError):
            return default

    weights = (num("wd", 0.6, 0, 1), num("wg", 0.3, 0, 1), num("wr", 0.1, 0, 1))
    recs = recommend_books(
        df, blocks, int(indices[key]), weights,
        top_n=num("n", 5, 1, 20, int),
        same_author=request.args.get("same_author") == "1",
        min_rating=num("min_rating", 0, 0, 5),
    )
    for r in recs:
        r["cover"] = covers.get(r["key"]) or None

    row = df.iloc[int(indices[key])]
    return jsonify(
        source={"key": key, "title": row["Display"], "author": row["Author"],
                "cover": covers.get(key) or None},
        results=recs,
    )


if __name__ == "__main__":
    app.run(debug=True)
