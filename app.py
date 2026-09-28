import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Title
st.title("🎬 Movie Recommender System")

# Description
st.markdown("""
**Discover your next favorite movie!**

Select a movie from the dropdown below, and our recommendation engine will find the top 5 most similar movies based on content similarity (genres, keywords, cast, crew, and overview).
""")

# Sample movie data (in production, this would come from a database or API)
@st.cache_data
def load_movie_data():
    """Load or create sample movie dataset"""
    movies = [
        {"id": 1, "title": "The Dark Knight", "genres": "Action, Crime, Drama"},
        {"id": 2, "title": "Inception", "genres": "Action, Adventure, Sci-Fi"},
        {"id": 3, "title": "Interstellar", "genres": "Adventure, Drama, Sci-Fi"},
        {"id": 4, "title": "The Matrix", "genres": "Action, Sci-Fi"},
        {"id": 5, "title": "Pulp Fiction", "genres": "Crime, Drama"},
        {"id": 6, "title": "Fight Club", "genres": "Drama"},
        {"id": 7, "title": "Forrest Gump", "genres": "Drama, Romance"},
        {"id": 8, "title": "The Shawshank Redemption", "genres": "Drama"},
        {"id": 9, "title": "The Godfather", "genres": "Crime, Drama"},
        {"id": 10, "title": "Goodfellas", "genres": "Crime, Drama"},
        {"id": 11, "title": "The Lord of the Rings: The Fellowship of the Ring", "genres": "Action, Adventure, Drama"},
        {"id": 12, "title": "Star Wars: Episode V - The Empire Strikes Back", "genres": "Action, Adventure, Fantasy"},
        {"id": 13, "title": "Back to the Future", "genres": "Adventure, Comedy, Sci-Fi"},
        {"id": 14, "title": "The Silence of the Lambs", "genres": "Crime, Drama, Thriller"},
        {"id": 15, "title": "Se7en", "genres": "Crime, Drama, Mystery"},
        {"id": 16, "title": "The Prestige", "genres": "Drama, Mystery, Sci-Fi"},
        {"id": 17, "title": "Gladiator", "genres": "Action, Adventure, Drama"},
        {"id": 18, "title": "The Departed", "genres": "Crime, Drama, Thriller"},
        {"id": 19, "title": "Whiplash", "genres": "Drama, Music"},
        {"id": 20, "title": "Parasite", "genres": "Comedy, Drama, Thriller"},
    ]
    return pd.DataFrame(movies)

# Load movie data
movies_df = load_movie_data()

# Create movie options for dropdown (title with ID for uniqueness)
movie_options = {f"{row['title']} ({row['genres']})": row['id'] for _, row in movies_df.iterrows()}

# Movie dropdown
st.subheader("Select a Movie")
selected_movie_label = st.selectbox(
    "Choose a movie you like:",
    options=list(movie_options.keys()),
    index=0,
    help="Start typing to search for a movie"
)

selected_movie_id = movie_options[selected_movie_label]
selected_movie = movies_df[movies_df['id'] == selected_movie_id].iloc[0]

# Display selected movie info
st.info(f"**Selected:** {selected_movie['title']} — *{selected_movie['genres']}*")

# Recommend button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_clicked = st.button(
        "🎯 Get Recommendations",
        type="primary",
        use_container_width=True
    )

# Simulated recommendation function (in production, this would use a real ML model)
@st.cache_data
def get_recommendations(movie_id, n=5):
    """Generate simulated recommendations with similarity scores"""
    # In a real app, this would compute cosine similarity on TF-IDF vectors
    # or use a pre-trained embedding model
    np.random.seed(movie_id)  # Deterministic for demo

    # Get other movies (exclude selected)
    other_movies = movies_df[movies_df['id'] != movie_id].copy()

    # Generate pseudo-similarity scores (higher for similar genres)
    selected_genres = set(movies_df[movies_df['id'] == movie_id]['genres'].values[0].split(', '))

    def compute_similarity(row):
        movie_genres = set(row['genres'].split(', '))
        overlap = len(selected_genres & movie_genres)
        base_score = 0.3 + overlap * 0.15
        noise = np.random.uniform(-0.05, 0.05)
        return min(0.98, max(0.45, base_score + noise))

    other_movies['similarity'] = other_movies.apply(compute_similarity, axis=1)
    other_movies = other_movies.sort_values('similarity', ascending=False).head(n)

    return other_movies[['id', 'title', 'genres', 'similarity']].reset_index(drop=True)

# Display recommendations
if recommend_clicked:
    with st.spinner("Finding similar movies..."):
        recommendations = get_recommendations(selected_movie_id, n=5)

    st.markdown("---")
    st.subheader("🎯 Top 5 Recommendations")

    # Display as cards in a grid
    cols = st.columns(5)

    for idx, (_, movie) in enumerate(recommendations.iterrows()):
        with cols[idx]:
            # Similarity score as percentage
            score_pct = int(movie['similarity'] * 100)

            # Color based on similarity
            if score_pct >= 80:
                score_color = "🟢"
            elif score_pct >= 65:
                score_color = "🟡"
            else:
                score_color = "🟠"

            st.markdown(f"""
            <div style="
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
                background-color: #fafafa;
                height: 100%;
            ">
                <h4 style="margin: 0 0 10px 0; color: #333;">{movie['title']}</h4>
                <p style="margin: 0 0 10px 0; color: #666; font-size: 0.85em;">{movie['genres']}</p>
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 8px;
                ">
                    <span style="font-size: 1.5em; font-weight: bold; color: #1f77b4;">{score_pct}%</span>
                    <span style="font-size: 1.2em;">{score_color}</span>
                </div>
                <div style="
                    width: 100%;
                    height: 8px;
                    background-color: #e0e0e0;
                    border-radius: 4px;
                    margin-top: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {score_pct}%;
                        height: 100%;
                        background: linear-gradient(90deg, #1f77b4, #4fc3f7);
                        border-radius: 4px;
                    "></div>
                </div>
                <p style="margin: 8px 0 0 0; font-size: 0.75em; color: #999;">Match Score</p>
            </div>
            """, unsafe_allow_html=True)

    # Also show as a clean table
    st.markdown("---")
    st.subheader("📊 Detailed Results")

    display_df = recommendations.copy()
    display_df['Similarity Score'] = display_df['similarity'].apply(lambda x: f"{x:.2%}")
    display_df = display_df.rename(columns={
        'title': 'Movie Title',
        'genres': 'Genres'
    })[['Movie Title', 'Genres', 'Similarity Score']]
    display_df.index = range(1, len(display_df) + 1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False
    )

    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"recommendations_for_{selected_movie['title'].replace(' ', '_')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.85em;">
    <p>Movie Recommender System • Built with Streamlit</p>
    <p>Similarity scores are simulated for demonstration purposes.</p>
</div>
""", unsafe_allow_html=True)