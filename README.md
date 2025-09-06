# Book Recommendation System

A book recommendation system designed for discovering similar books from different authors. If you loved Harry Potter, this will help you find similar books by authors other than J.K. Rowling!

## Demo

[Live Demo](your-deployment-link-here) *(coming soon)*

## About The Project

I created this recommendation system because reading is one of my favorite hobbies, and I wanted to solve a specific problem: finding books similar to ones I love, but by different authors. It's easy to search for more books by the same author, but discovering similar stories by different writers is much harder.

The system analyzes books based on:
- **Genres** - Fiction, fantasy, romance, etc.
- **Descriptions** - Plot summaries and themes
- **Average ratings** - Quality indicators from Goodreads

## Built With

- **Python** - Core recommendation logic
- **Flask** - Web framework
- **HTML/CSS** - Frontend interface
- **scikit-learn** - Machine learning (KNN with cosine distance)
- **Pandas & NumPy** - Data processing

## How It Works

The system uses K-Nearest Neighbors (KNN) with cosine distance on matrices representing different book features. To optimize performance, I pre-compute and cache all matrices instead of calculating them on each search request.

**Dataset**: ~10,000 books from Goodreads

## Getting Started

### Prerequisites

- Python 3.7+
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

3. **Set up the cache matrices** (one-time setup)
   ```bash
   python setup_cache.py
   ```
   *This will create pre-computed matrices in the `data/matrices/` folder*

4. **Run the Flask application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```


## Usage

1. Enter a book title in the search box
2. The system finds books with similar genres, descriptions, and ratings
3. Results show books by **different authors** to help you discover new writers
4. You can then find recommendations by different weights.

## ⚡ Optimizations

**Problem**: Computing similarity matrices for each search was slow (~4 seconds per request)

**Solution**: Pre-compute all matrices once during setup and cache them as pickle files. This reduced search time to milliseconds while only requiring one initial computation instead of four per search.

## Future Improvements

- Add more sophisticated NLP for book descriptions
- Include user ratings and collaborative filtering
- Expand dataset to 100k+ books
- Add book cover images
- Implement user accounts and reading lists

## Lessons Learned

This was my first real dive into recommendation systems and web development. While the approach isn't the most sophisticated (there are definitely better algorithms out there), it works well for my specific use case and taught me a lot about:

- Feature engineering for text data
- Performance optimization through caching
- Flask web development basics
- The importance of understanding your specific problem before jumping into complex solutions


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dataset

Using a subset of the Goodreads dataset (~10,000 books) for performance and simplicity.