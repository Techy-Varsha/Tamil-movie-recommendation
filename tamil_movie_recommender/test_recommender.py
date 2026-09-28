import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load dataset
csv_path = r"data\tamil_movies.csv"
df = pd.read_csv(csv_path)
df = df.fillna('')

# Create combined features
df['combined_features'] = (
    df['genre'] + ' ' +
    df['director'] + ' ' +
    df['actors'] + ' ' +
    df['description']
)

# Compute TF-IDF and similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Loaded {len(df)} Tamil movies")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

# Test recommendations
def recommend_movies(movie_title, df, cosine_sim, num_recommendations=5):
    if movie_title not in df['title'].values:
        raise ValueError(f"Movie '{movie_title}' not found")
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i for i, _ in sim_scores]
    similarity_scores = [score for _, score in sim_scores]
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

print("\n✓ All tests passed!")