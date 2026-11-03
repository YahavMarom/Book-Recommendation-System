import ast
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix  # noqa: F401  (kept for callers)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


# ---------------------------------------------------------------- cleaning
def remove_parens(x):
    return re.sub(r"[\(\[].*?[\)\]]", "", x).strip()


def is_english(text):
    if re.fullmatch(r"[A-Za-z0-9\s'\":;,\-!?.]+", text):
        return True
    try:
        return detect(str(text)) == "en"
    except Exception:
        return False


def parse_genres(x):
    """CSV stores genres as the string "['Classics', 'Fiction']" -> real list."""
    try:
        v = ast.literal_eval(x)
        return [str(g).strip().lower() for g in v] if isinstance(v, list) else []
    except Exception:
        return []


def load_data():
    df = pd.read_csv("data/data.csv", encoding="utf-8-sig")

    df = df[df["Book"].apply(is_english)]
    df["Description"] = df["Description"].fillna("")
    df["Genres"] = df["Genres"].fillna("")
    df["Avg_Rating"] = pd.to_numeric(df["Avg_Rating"], errors="coerce")
    df["Num_Ratings"] = df["Num_Ratings"].replace(",", "", regex=True)
    df["Num_Ratings"] = pd.to_numeric(df["Num_Ratings"], errors="coerce").fillna(0).astype(int)

    df = df[df["Num_Ratings"] >= 10].reset_index(drop=True)

    # NEW: keep the original-case title for display, lowercase one is the lookup key
    df["Display"] = df["Book"].apply(remove_parens)
    df["Book"] = df["Display"].str.lower()
    df = df.drop_duplicates(subset=["Book"], keep="first").reset_index(drop=True)

    df["Genre_List"] = df["Genres"].apply(parse_genres)
    return df


def create_indices(df):
    return pd.Series(df.index, index=df["Book"])


# ---------------------------------------------------------------- features
# Stronger (slower, ~3-5x) alternatives: "all-mpnet-base-v2", "BAAI/bge-base-en-v1.5"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_DESC = 0.30   # minimum description similarity (tuned for MiniLM; retune if you change model)


def create_description_matrix(df, model_name=MODEL_NAME):
    """Sentence embeddings instead of TF-IDF (matches meaning, not just words)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # Description only: a short title ("...Man...") otherwise dominates the vector.
    # Title is used only as a fallback when the description is missing/very short.
    texts = [d if len(d) > 40 else f"{t}. {d}"
             for t, d in zip(df["Display"], df["Description"])]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    return emb.astype(np.float32)


def create_genres_matrix(df):
    """Binary genre tags, TF-IDF weighted so 'Fiction' counts less than 'Dystopia'."""
    cv = CountVectorizer(analyzer=lambda x: x, binary=True)
    counts = cv.fit_transform(df["Genre_List"])
    return TfidfTransformer(sublinear_tf=True).fit_transform(counts).tocsr()


def create_rating_matrix(df):
    """
    Fixed rating block: log-transform popularity, then *standardize* (center)
    both columns. Without centering every book points the same way and the
    block carries almost no information.
    """
    rating = df["Avg_Rating"].fillna(df["Avg_Rating"].mean()).to_numpy()
    pop = np.log1p(df["Num_Ratings"].to_numpy())
    feats = np.column_stack([rating, pop]).astype(np.float64)
    feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
    return feats.astype(np.float32)


def build_blocks(D, G, R):
    """Description and genre blocks are L2-normalized (cosine). The rating block is
    kept as raw z-scores: it is compared by distance, not cosine (see recommend_books).
    The total is still  wd*sim_desc + wg*sim_genre + wr*sim_rating, so the
    'why this book' split stays exact."""
    return normalize(D), normalize(G), R


# ---------------------------------------------------------------- recommend
def recommend_books(df, blocks, idx, weights=(0.6, 0.3, 0.1), top_n=5,
                    same_author=False, min_rating=0.0, min_desc=MIN_DESC):
    Dn, Gn, Rn = blocks

    w = np.clip(np.asarray(weights, dtype=float), 0, None)
    w = w / w.sum() if w.sum() > 0 else np.array([0.6, 0.3, 0.1])
    wd, wg, wr = w

    sd = Dn @ Dn[idx]
    sg = (Gn @ Gn[idx].T).toarray().ravel()
    # rating/popularity closeness in [0,1]: 1 = same quality & popularity level
    sr = np.clip(1 - np.abs(Rn - Rn[idx]).mean(axis=1) / 3, 0, 1)

    total = wd * sd + wg * sg + wr * sr
    total[idx] = -np.inf

    src_genres = set(df.at[idx, "Genre_List"])
    src_author = df.at[idx, "Author"]
    # unticked: hide the searched book's author AND avoid repeats among results
    seen_authors, result = {src_author}, []

    for j in np.argsort(-total)[:300]:
        row = df.iloc[j]
        if row["Avg_Rating"] < min_rating:
            continue
        if wd >= 0.2 and sd[j] < min_desc:   # too different in content: show fewer, not worse
            continue
        if not same_author and row["Author"] in seen_authors:
            continue
        seen_authors.add(row["Author"])

        parts = np.clip([wd * sd[j], wg * sg[j], wr * sr[j]], 0, None)
        shares = parts / parts.sum() if parts.sum() > 0 else parts

        result.append({
            "key": row["Book"],
            "title": row["Display"],
            "author": row["Author"],
            "rating": float(row["Avg_Rating"]),
            "similarity": round(float(total[j]), 3),
            "why": {
                "description": round(float(shares[0]) * 100),
                "genres": round(float(shares[1]) * 100),
                "rating": round(float(shares[2]) * 100),
            },
            "shared_genres": sorted(src_genres & set(row["Genre_List"]))[:6],
        })
        if len(result) >= top_n:
            break
    return result