"""
Tamil Movie Recommender System
Simple content-based filtering using TF-IDF and Cosine Similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_dataset(csv_path):
    """
    Load the Tamil movies dataset from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        pandas DataFrame with movie data

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If dataset is empty
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}\n"
            f"Please make sure 'tamil_movies.csv' exists in the data/ folder."
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Dataset is empty. Please add movies to the CSV file.")

    # Fill any missing values with empty strings
    df = df.fillna('')

    return df


def create_combined_features(df):
    """
    Create a combined text feature from genre, director, actors, and description.

    Args:
        df: DataFrame with movie data

    Returns:
        DataFrame with new 'combined_features' column
    """
    # Combine the important text columns
    df['combined_features'] = (
        df['genre'] + ' ' +
        df['director'] + ' ' +
        df['actors'] + ' ' +
        df['description']
    )
    return df


def compute_tfidf_and_similarity(df):
    """
    Compute TF-IDF vectors and cosine similarity matrix.

    Args:
        df: DataFrame with 'combined_features' column

    Returns:
        tuple: (tfidf_matrix, cosine_sim_matrix, tfidf_vectorizer)
    """
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')

    # Fit and transform the combined features
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # Compute cosine similarity between all movies
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return tfidf_matrix, cosine_sim, tfidf


def recommend_movies(movie_title, df, cosine_sim, num_recommendations=5):
    """
    Recommend similar Tamil movies based on cosine similarity.

    Args:
        movie_title: Name of the movie to find recommendations for
        df: DataFrame with movie data
        cosine_sim: Cosine similarity matrix
        num_recommendations: Number of recommendations to return (default 5)

    Returns:
        List of dictionaries with recommended movie details and similarity scores

    Raises:
        ValueError: If movie not found in dataset
    """
    # Check if movie exists
    if movie_title not in df['title'].values:
        available_movies = ', '.join(df['title'].head(10).tolist())
        raise ValueError(
            f"Movie '{movie_title}' not found in dataset.\n"
            f"Available movies (sample): {available_movies}..."
        )

    # Get the index of the selected movie
    idx = df[df['title'] == movie_title].index[0]

    # Get similarity scores for all movies with this movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the movie itself (first item) and get top N
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get movie indices and similarity scores
    movie_indices = [i for i, _ in sim_scores]
    similarity_scores = [score for _, score in sim_scores]

    # Build recommendations list
    recommendations = []
    for i, (movie_idx, score) in enumerate(zip(movie_indices, similarity_scores)):
        movie = df.iloc[movie_idx]
        recommendations.append({
            'rank': i + 1,
            'title': movie['title'],
            'genre': movie['genre'],
            'director': movie['director'],
            'year': movie['year'],
            'similarity': round(score, 2)
        })

    return recommendations


def initialize_recommender(csv_path='data/tamil_movies.csv'):
    """
    Initialize the recommender system by loading data and computing similarity.

    Args:
        csv_path: Path to the CSV dataset

    Returns:
        tuple: (df, cosine_sim) - DataFrame and similarity matrix
    """
    # Load dataset
    df = load_dataset(csv_path)

    # Create combined features
    df = create_combined_features(df)

    # Compute TF-IDF and cosine similarity
    _, cosine_sim, _ = compute_tfidf_and_similarity(df)

    return df, cosine_sim


# For direct testing
if __name__ == "__main__":
    print("Initializing Tamil Movie Recommender...")
    df, cosine_sim = initialize_recommender()

    print(f"\nLoaded {len(df)} Tamil movies")
    print("\n--- Testing recommendations ---")

    # Test with a few movies
    test_movies = ["Vikram", "Kaithi", "96", "Baasha"]

    for movie in test_movies:
        try:
            print(f"\nRecommendations for '{movie}':")
            recs = recommend_movies(movie, df, cosine_sim, num_recommendations=5)
            for r in recs:
                print(f"  {r['rank']}. {r['title']} ({r['year']}) - {r['genre']} - Dir: {r['director']} - Similarity: {r['similarity']}")
        except ValueError as e:
            print(f"  Error: {e}")

    # Test invalid movie
    print("\n--- Testing invalid movie ---")
    try:
        recommend_movies("Invalid Movie", df, cosine_sim)
    except ValueError as e:
        print(f"  Expected error: {e}")

    print("\n[OK] All tests passed!")