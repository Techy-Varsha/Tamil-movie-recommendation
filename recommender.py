"""
Movie Recommender System using Content-Based Filtering

This module provides a movie recommendation system based on TF-IDF vectorization
and cosine similarity of movie features (genres, keywords, overview, etc.).
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class MovieRecommender:
    """
    A content-based movie recommender using TF-IDF and cosine similarity.

    Attributes:
        movies_df: DataFrame containing movie data
        tfidf_matrix: TF-IDF feature matrix
        cosine_sim: Cosine similarity matrix
        tfidf_vectorizer: Fitted TF-IDF vectorizer
        indices: Mapping of movie titles to DataFrame indices
    """

    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        """
        Initialize the MovieRecommender.

        Args:
            max_features: Maximum number of TF-IDF features
            stop_words: Stop words to remove ('english' or None)
        """
        self.max_features = max_features
        self.stop_words = stop_words
        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.tfidf_vectorizer = None
        self.indices = None

    def load_dataset(self, filepath: str, title_col: str = 'title',
                     feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Load and prepare the movie dataset.

        Args:
            filepath: Path to the CSV file
            title_col: Column name for movie titles
            feature_cols: List of columns to combine for features
                          (default: ['genres', 'keywords', 'overview', 'tagline', 'cast', 'crew'])

        Returns:
            Prepared DataFrame with combined features
        """
        self.movies_df = pd.read_csv(filepath)

        if feature_cols is None:
            feature_cols = ['genres', 'keywords', 'overview', 'tagline', 'cast', 'crew']

        # Keep only available columns
        available_cols = [col for col in feature_cols if col in self.movies_df.columns]

        if not available_cols:
            raise ValueError(f"No feature columns found. Available: {list(self.movies_df.columns)}")

        # Fill NaN values with empty string
        for col in available_cols:
            self.movies_df[col] = self.movies_df[col].fillna('')

        # Combine features into a single text column
        self.movies_df['combined_features'] = self.movies_df[available_cols].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1
        )

        # Create title-to-index mapping
        self.indices = pd.Series(
            self.movies_df.index,
            index=self.movies_df[title_col]
        ).drop_duplicates()

        return self.movies_df

    def fit(self) -> 'MovieRecommender':
        """
        Fit the TF-IDF vectorizer and compute cosine similarity matrix.

        Returns:
            Self for method chaining
        """
        if self.movies_df is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        # Initialize and fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=(1, 2)
        )

        # Transform combined features
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movies_df['combined_features']
        )

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        return self

    def recommend_movies(self, title: str, top_n: int = 10,
                         min_score: float = 0.0) -> pd.DataFrame:
        """
        Get movie recommendations based on cosine similarity.

        Args:
            title: Movie title to find recommendations for
            top_n: Number of recommendations to return
            min_score: Minimum similarity score threshold (0.0 to 1.0)

        Returns:
            DataFrame with recommended movies and similarity scores

        Raises:
            KeyError: If movie title not found in dataset
        """
        if self.cosine_sim is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if title not in self.indices:
            available_titles = list(self.indices.index)[:10]
            raise KeyError(
                f"Movie '{title}' not found. "
                f"Available titles (sample): {available_titles}..."
            )

        # Get index of the movie
        idx = self.indices[title]

        # Get similarity scores for all movies
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort by similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the movie itself (first item) and apply threshold
        sim_scores = [(i, score) for i, score in sim_scores[1:] if score >= min_score]

        # Get top N recommendations
        top_recommendations = sim_scores[:top_n]

        # Create results DataFrame
        movie_indices = [i for i, _ in top_recommendations]
        similarity_scores = [score for _, score in top_recommendations]

        results = self.movies_df.iloc[movie_indices].copy()
        results['similarity_score'] = similarity_scores

        # Select relevant columns for output
        output_cols = ['title', 'similarity_score']
        for col in ['genres', 'overview', 'vote_average', 'release_date']:
            if col in results.columns:
                output_cols.append(col)

        return results[output_cols].reset_index(drop=True)

    def get_similar_movies_by_index(self, idx: int, top_n: int = 10) -> pd.DataFrame:
        """
        Get similar movies by DataFrame index.

        Args:
            idx: DataFrame index of the movie
            top_n: Number of recommendations to return

        Returns:
            DataFrame with recommended movies and similarity scores
        """
        if self.cosine_sim is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if idx >= len(self.movies_df):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.movies_df)}")

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]

        movie_indices = [i for i, _ in sim_scores]
        similarity_scores = [score for _, score in sim_scores]

        results = self.movies_df.iloc[movie_indices].copy()
        results['similarity_score'] = similarity_scores

        return results

    def get_feature_names(self) -> List[str]:
        """Get the TF-IDF feature names (vocabulary)."""
        if self.tfidf_vectorizer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.tfidf_vectorizer.get_feature_names_out().tolist()

    def save_model(self, filepath: str) -> None:
        """Save the fitted model components."""
        import pickle
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'movies_df': self.movies_df,
            'max_features': self.max_features,
            'stop_words': self.stop_words
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load_model(cls, filepath: str) -> 'MovieRecommender':
        """Load a saved model."""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        recommender = cls(
            max_features=model_data['max_features'],
            stop_words=model_data['stop_words']
        )
        recommender.tfidf_vectorizer = model_data['tfidf_vectorizer']
        recommender.tfidf_matrix = model_data['tfidf_matrix']
        recommender.cosine_sim = model_data['cosine_sim']
        recommender.indices = model_data['indices']
        recommender.movies_df = model_data['movies_df']

        return recommender


def recommend_movies(title: str, movies_df: pd.DataFrame,
                     tfidf_matrix, cosine_sim, indices,
                     top_n: int = 10, min_score: float = 0.0) -> pd.DataFrame:
    """
    Standalone function to recommend movies (for simple use cases).

    Args:
        title: Movie title to find recommendations for
        movies_df: DataFrame containing movie data
        tfidf_matrix: Pre-computed TF-IDF matrix
        cosine_sim: Pre-computed cosine similarity matrix
        indices: Title-to-index mapping (pd.Series)
        top_n: Number of recommendations to return
        min_score: Minimum similarity score threshold

    Returns:
        DataFrame with recommended movies and similarity scores
    """
    if title not in indices:
        raise KeyError(f"Movie '{title}' not found in dataset")

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, score) for i, score in sim_scores[1:] if score >= min_score]
    top_recommendations = sim_scores[:top_n]

    movie_indices = [i for i, _ in top_recommendations]
    similarity_scores = [score for _, score in top_recommendations]

    results = movies_df.iloc[movie_indices].copy()
    results['similarity_score'] = similarity_scores

    return results


# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = {
        'title': [
            'The Dark Knight', 'Inception', 'The Matrix', 'Interstellar',
            'Batman Begins', 'The Prestige', 'Memento', 'Tenet',
            'Avatar', 'Titanic', 'The Avengers', 'Iron Man',
            'Spider-Man', 'Captain America', 'Thor', 'Guardians of the Galaxy'
        ],
        'genres': [
            'Action Crime Drama', 'Action Sci-Fi Thriller', 'Action Sci-Fi',
            'Adventure Drama Sci-Fi', 'Action Adventure', 'Drama Mystery Sci-Fi',
            'Mystery Thriller', 'Action Sci-Fi Thriller', 'Action Adventure Fantasy',
            'Drama Romance', 'Action Adventure Sci-Fi', 'Action Adventure Sci-Fi',
            'Action Adventure', 'Action Adventure Sci-Fi', 'Action Adventure Fantasy',
            'Action Adventure Comedy Sci-Fi'
        ],
        'keywords': [
            'batman joker gotham', 'dream heist subconscious', 'reality simulation',
            'space time blackhole', 'batman origin gotham', 'magic rivalry',
            'memory tattoo', 'time inversion', 'pandora alien', 'ship romance iceberg',
            'superhero team', 'technology armor', 'spider powers', 'super soldier',
            'norse god', 'space team raccoon'
        ],
        'overview': [
            'Batman faces the Joker in Gotham', 'Thief enters dreams to plant ideas',
            'Hacker discovers simulated reality', 'Explorers travel through wormhole',
            'Bruce Wayne becomes Batman', 'Two magicians bitter rivalry',
            'Man with memory loss seeks revenge', 'Agent manipulates time flow',
            'Marine on alien world', 'Love story on doomed ship',
            'Heroes unite against threat', 'Billionaire builds armored suit',
            'Teen gets spider powers', 'Soldier becomes super hero',
            'God of thunder on Earth', 'Misfits save galaxy'
        ],
        'tagline': [
            'Why so serious?', 'Your mind is the scene of the crime',
            'What is the Matrix?', 'Mankind was born on Earth',
            'Evil fears the knight', 'Are you watching closely?',
            'Some memories are best forgotten', 'Time runs out',
            'Enter the world of Pandora', 'Nothing on Earth could come between them',
            'Avengers assemble', 'Heroes aren\'t born, they\'re built',
            'With great power comes great responsibility',
            'The first Avenger', 'Two worlds, one hero',
            'All heroes start somewhere'
        ],
        'cast': [
            'Christian Bale Heath Ledger', 'Leonardo DiCaprio Marion Cotillard',
            'Keanu Reeves Laurence Fishburne', 'Matthew McConaughey Anne Hathaway',
            'Christian Bale Michael Caine', 'Christian Bale Hugh Jackman',
            'Guy Pearce Carrie-Anne Moss', 'John David Washington Robert Pattinson',
            'Sam Worthington Zoe Saldana', 'Leonardo DiCaprio Kate Winslet',
            'Robert Downey Jr Chris Evans', 'Robert Downey Jr Gwyneth Paltrow',
            'Tom Holland Zendaya', 'Chris Evans Hayley Atwell',
            'Chris Hemsworth Natalie Portman', 'Chris Pratt Zoe Saldana'
        ],
        'crew': [
            'Christopher Nolan', 'Christopher Nolan', 'Lana Wachowski',
            'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan',
            'Christopher Nolan', 'Christopher Nolan', 'James Cameron',
            'James Cameron', 'Joss Whedon', 'Jon Favreau', 'Jon Watts',
            'Joe Russo', 'Kenneth Branagh', 'James Gunn'
        ],
        'vote_average': [9.0, 8.8, 8.7, 8.6, 8.2, 8.5, 8.4, 7.5, 7.8, 7.9, 8.0, 7.9, 7.5, 6.9, 7.0, 8.0],
        'release_date': [
            '2008-07-18', '2010-07-16', '1999-03-31', '2014-11-07',
            '2005-06-15', '2006-10-20', '2000-10-11', '2020-08-26',
            '2009-12-18', '1997-12-19', '2012-05-04', '2008-05-02',
            '2021-12-17', '2011-07-22', '2011-05-06', '2014-08-01'
        ]
    }

    # Create DataFrame and save to CSV
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_movies.csv', index=False)
    print("Sample dataset created: sample_movies.csv")

    # Initialize and use recommender
    recommender = MovieRecommender(max_features=500, stop_words='english')
    recommender.load_dataset('sample_movies.csv')
    recommender.fit()

    # Get recommendations
    print("\n--- Recommendations for 'The Dark Knight' ---")
    recommendations = recommender.recommend_movies('The Dark Knight', top_n=5)
    print(recommendations.to_string(index=False))

    print("\n--- Recommendations for 'Inception' ---")
    recommendations = recommender.recommend_movies('Inception', top_n=5)
    print(recommendations.to_string(index=False))

    print("\n--- Recommendations for 'The Matrix' ---")
    recommendations = recommender.recommend_movies('The Matrix', top_n=5)
    print(recommendations.to_string(index=False))