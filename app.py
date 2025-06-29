# app.py

import streamlit as st
import ast  # for safely parsing the genres field

from utils import (
    load_data,
    generate_tfidf_and_similarity,
    hybrid_recommendation,
    fetch_poster
)

# ⛓️ Set your TMDB API key
TMDB_API_KEY = "f2b4381c47b1e81c103225bdb5e41a22"  # 🔐 Replace with your real API key

# 🎬 App title
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommendation System")

# 📦 Load data
movies, ratings = load_data()
similarity_matrix = generate_tfidf_and_similarity(movies)

# 🎯 Movie selection input
movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie to get recommendations:", sorted(movie_list))

# 🔍 Recommendation button
if st.button("Recommend"):
    recommendations = hybrid_recommendation(selected_movie, movies, similarity_matrix, top_n=5)

    if not recommendations.empty:
        st.subheader(f"📽️ Top 5 recommendations for '{selected_movie}'")

        cols = st.columns(5)
        for i, row in recommendations.iterrows():
            poster_url = fetch_poster(row['title'], TMDB_API_KEY)

            # Safely extract genres
            try:
                genre_list = ast.literal_eval(row['genres'])
                genre_str = ", ".join([g['name'] for g in genre_list])
            except:
                genre_str = "Unknown"

            with cols[i]:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                st.markdown(f"**{row['title']}**")
                st.caption(
                    f"⭐ {row['vote_average']:.1f} &nbsp;&nbsp;💬 {row['vote_count']:,} votes<br>🎞️ {genre_str}",
                    unsafe_allow_html=True
                )
    else:
        st.warning("Sorry, we couldn't find similar movies.")
