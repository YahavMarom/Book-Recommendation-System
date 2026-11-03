



# Book Recommendation System

A book recommendation system designed for discovering similar books from different authors. 

## Demo

![Demo]([images/demo.mp4](https://github.com/user-attachments/assets/761bcf2e-efb7-49a3-ba7f-6f069c0b2e81))


## About The Project

I created this recommendation system because reading is one of my favorite hobbies, and I wanted to solve a specific problem: finding books similar to ones I love, but by different authors. It's easy to search for more books by the same author, but discovering similar stories by different writers is much harder.

The system compares books using:
- **Descriptions** - plot summaries and themes, compared by meaning using sentence embeddings
- **Genres** - Goodreads genre tags, weighted so that broad tags like "Fiction" count for less
- **Ratings** - average rating and number of ratings, used as a quality and popularity signal

Each result shows a cover, the author, the rating, and a bar showing how much of the match came from the description, the genres and the rating. The weights can be changed with sliders, and the results update immediately.

## Built With

- **Python** - core recommendation logic
- **Flask** - web framework and JSON API
- **JavaScript / HTML / CSS** - interface, autocomplete and live sliders (no framework)
- **sentence-transformers** - description embeddings (`all-MiniLM-L6-v2`)
- **scikit-learn** - genre weighting and normalization
- **Pandas, NumPy & SciPy** - data processing
- **Open Library API** - book covers

## How It Works

Each book is represented by three blocks: a description embedding, a TF-IDF weighted genre vector, and a rating/popularity score. Description and genre similarity use cosine similarity. Rating similarity is based on the distance between two books' rating and popularity (log of the number of ratings).

The final score is a weighted sum of the three:

```
score = w_description * sim_description + w_genres * sim_genres + w_rating * sim_rating
```

Because the score is a plain sum, each result can show exactly how much each part contributed. That is where the "why this book" bar comes from. Results with a very low description similarity are dropped, and by default the searched book's author and repeated authors are left out.

Everything slow (embeddings, genre matrix, cover lookups) is computed once in `setup_cache.py`. A search only does a few vector operations over the saved matrices.

**Dataset**: about 8,700 books from Goodreads (after removing non-English titles and books with fewer than 10 ratings)

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yahavmarom/book-recommendation-system.git
   cd book-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the cache** (one-time setup)
   ```bash
   python setup_cache.py
   ```
   *This computes the embeddings and matrices in `data/matrices/` and looks up cover URLs from Open Library into `data/covers.json`. The covers take a long time (it took me several hours for the full dataset), but progress is saved and the script resumes where it stopped.*

   To rebuild only the embeddings and matrices without touching the covers:
   ```bash
   python setup_cache.py --skip-covers
   ```

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

## Usage

1. Start typing a book title. Matching titles appear as you type (titles starting with your text come first). Use the arrow keys and Enter, or click.
2. The system shows similar books with covers, authors and ratings.
3. Move the **Description / Genres / Rating** sliders to change how books are compared. The results update as you drag.
4. Set the number of results, a minimum rating, or include the same author.
5. Click any recommendation to search from that book. The URL (`?book=...`) can be shared.

## Optimizations

**Problem**: Computing the similarity matrices for every search was slow (about 4 seconds per request in the first version).

**Solution**: Compute the matrices once during setup and save them to disk. The app loads them at startup, so a search is only a few matrix-vector products. The same idea applies to covers (looked up once, stored as URLs) and to autocomplete (the whole title list is loaded in the browser once and filtered locally).

## Possible Improvements 

- Search by description ("man contemplates meaning of life under the verge of world war I") using the existing embeddings
- Re-ranking to avoid near-duplicate results (series, editions)
- Expand the dataset to 100k+ books

## Lessons Learned

This started as my first dive into recommendation systems and web development. The first version used TF-IDF and KNN, and the second version taught me where it was weak:

- TF-IDF matches words, not meanings. Sentence embeddings find related books that share no vocabulary.
- Short titles can dominate an embedding, so descriptions are embedded without the title.
- Caching slow work in a setup step keeps the app fast.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dataset

Using a subset of the Goodreads dataset (about 8,700 books after filtering) for performance and simplicity. Cover images come from Open Library.
