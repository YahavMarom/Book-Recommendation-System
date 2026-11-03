"""Run once:  python setup_cache.py   (re-run after changing data.csv or the model)

Builds embeddings + genre/rating blocks and looks up cover URLs.
Cover lookups are cached in data/covers.json, so an interrupted run resumes.
"""
import argparse
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse

from recommendations import (load_data, create_description_matrix,
                             create_genres_matrix, create_rating_matrix)

os.makedirs("data/matrices", exist_ok=True)
COVERS_PATH = "data/covers.json"


def fetch_cover(title, author):
    """Open Library search -> cover id -> image URL. Returns None if not found."""
    q = urllib.parse.urlencode({"title": title, "author": author, "limit": 1,
                                "fields": "cover_i"})
    url = f"https://openlibrary.org/search.json?{q}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "book-recs-student-project"})
            with urllib.request.urlopen(req, timeout=15) as r:
                docs = json.load(r).get("docs", [])
            if docs and docs[0].get("cover_i"):
                return f"https://covers.openlibrary.org/b/id/{docs[0]['cover_i']}-M.jpg"
            return ""          # searched, no cover (store "" so we don't retry)
        except Exception:
            time.sleep(1 + attempt)
    return None                # network failure -> retry on next run


def main(skip_covers=False):
    df = load_data()
    print(f"{len(df)} books")

    np.save("data/matrices/desc.npy", create_description_matrix(df))
    sparse.save_npz("data/matrices/genre.npz", create_genres_matrix(df))
    np.save("data/matrices/rating.npy", create_rating_matrix(df))
    pickle.dump(df, open("data/matrices/dataframe.pkl", "wb"))

    # autocomplete list: key (lowercase), display title, author
    titles = [{"k": k, "t": t, "a": a}
              for k, t, a in zip(df["Book"], df["Display"], df["Author"])]
    json.dump(titles, open("data/book_titles.json", "w"))

    if skip_covers:
        print("skipping covers (using existing data/covers.json)")
        return

    # covers (polite: 4 threads, resumable)
    covers = json.load(open(COVERS_PATH)) if os.path.exists(COVERS_PATH) else {}
    todo = [(k, t, a) for k, t, a in zip(df["Book"], df["Display"], df["Author"])
            if covers.get(k) is None]
    print(f"{len(todo)} covers to look up")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for n, ((k, _, _), res) in enumerate(
                zip(todo, ex.map(lambda x: fetch_cover(x[1], x[2]), todo)), 1):
            if res is not None:
                covers[k] = res
            if n % 200 == 0:
                json.dump(covers, open(COVERS_PATH, "w"))
                print(f"  {n}/{len(todo)}")
    json.dump(covers, open(COVERS_PATH, "w"))
    print("done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-covers", action="store_true",
                    help="rebuild embeddings/matrices only; never touch covers")
    main(ap.parse_args().skip_covers)