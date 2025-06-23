# app.py

import streamlit as st
from utils import (
    load_data,
    generate_tfidf_and_similarity,
    hybrid_recommendation,
    fetch_poster
)

# ⛓️ Set your TMDB API key
TMDB_API_KEY = st.secrets["tmdb"]["api_key"]  # 🔐 Replace with your real API key

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
            with cols[i]:
                if poster_url:
                    st.image(poster_url, use_container_width=True)
            st.markdown(f"**{row['title']}**")
            st.caption(f"⭐ {row['vote_average']}  | 🗳️ {row['vote_count']} votes")
    else:
        st.warning("Sorry, we couldn't find similar movies.")
