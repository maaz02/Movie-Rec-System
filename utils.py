# utils.py

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

@st.cache_data
def load_data(movies_path='data/movie.csv', ratings_path='data/rating.csv'):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

@st.cache_resource
def generate_tfidf_and_similarity(movies):
    # Prepare TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])

    # Compute similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity_matrix

def hybrid_recommendation(movie_title, movies, similarity_matrix, top_n=10):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    if movie_title not in indices:
        return pd.DataFrame()

    idx = indices[movie_title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]
    return movies[['title', 'vote_average', 'vote_count']].iloc[movie_indices].reset_index(drop=True)

def fetch_poster(title, api_key):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            return full_path
    except:
        return None
