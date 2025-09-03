# The Project
I created a book recommendation system, created for discovering similar books from different authors.

## How
Tech: Python, HTML, Flask

I wanted to create a recommendation system, but not a complicated one. Reading books is a favorite hobby of mine, and creating this rec-system was a cool idea. 
The things I had in mine were recommending books based on various factors - genres, description and average rating (from goodreads dataset). I took a small (10k~) dataset to run my project on. I creates matrices that describe the factors above, and ran KNN with cosine distance. It is of course not the optimal approach, but it works, and I can always improve it. 
For my purposes, I want to have several books, all by distinct authors. Because if I liked Harry Potter, it's easy for me to search up books by Rowling. However, I might have trouble finding similar books by different authors. 
