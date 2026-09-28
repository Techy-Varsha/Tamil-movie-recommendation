"""
Tamil Movie Recommendation System - Streamlit Web App
Simple content-based movie recommender using TF-IDF and Cosine Similarity
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path to import recommender
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender import (
    load_dataset,
    create_combined_features,
    compute_tfidf_and_similarity,
    recommend_movies
)


# Page configuration
st.set_page_config(
    page_title="Tamil Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)


@st.cache_data
def initialize_system():
    """Load dataset and compute similarity matrix (cached for performance)."""
    csv_path = 'data/tamil_movies.csv'

    # Load dataset
    df = load_dataset(csv_path)

    # Create combined features
    df = create_combined_features(df)

    # Compute TF-IDF and cosine similarity
    _, cosine_sim, _ = compute_tfidf_and_similarity(df)

    return df, cosine_sim


def main():
    # Title
    st.title("🎬 Tamil Movie Recommendation System")

    # Description
    st.markdown("""
    **Select a Tamil movie and discover similar movies!**

    This system uses *content-based filtering* with TF-IDF vectorization and cosine similarity
    to recommend movies based on their genre, director, actors, and description.
    """)

    # Initialize the recommender system
    try:
        with st.spinner("Loading movie database..."):
            df, cosine_sim = initialize_system()

        st.success(f"✓ Loaded {len(df)} Tamil movies")

    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the system: {e}")
        st.stop()

    # Movie selection dropdown
    movie_titles = sorted(df['title'].tolist())

    selected_movie = st.selectbox(
        "Select a movie:",
        options=movie_titles,
        index=0,
        placeholder="Choose a movie..."
    )

    # Recommend button
    if st.button("🎯 Recommend Movies", type="primary", use_container_width=True):
        if selected_movie:
            with st.spinner("Finding similar movies..."):
                try:
                    recommendations = recommend_movies(
                        selected_movie,
                        df,
                        cosine_sim,
                        num_recommendations=5
                    )

                    # Display results
                    st.markdown("---")
                    st.subheader("🎬 Recommended Movies")

                    st.markdown("""
                    *These movies are recommended because their genre, director, actors,
                    and description are similar to the selected movie.*
                    """)

                    # Display each recommendation
                    for rec in recommendations:
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{rec['rank']}. {rec['title']}** ({rec['year']})")
                                st.caption(f"Genre: {rec['genre']}  |  Director: {rec['director']}")
                            with col2:
                                # Color-coded similarity
                                sim = rec['similarity']
                                if sim >= 0.5:
                                    color = "green"
                                elif sim >= 0.3:
                                    color = "orange"
                                else:
                                    color = "red"
                                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.2em;'>Similarity: {sim:.2f}</span>", unsafe_allow_html=True)
                            st.divider()

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please select a movie from the dropdown.")

    # Footer
    st.markdown("---")
    st.caption("""
    **How it works:** Movie metadata → Combined text features → TF-IDF Vectorization → Cosine Similarity → Top 5 similar movies
    """)


if __name__ == "__main__":
    main()