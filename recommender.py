import streamlit as st

# ✅ Set page config FIRST
st.set_page_config(layout="wide")

import pandas as pd
import pickle
import requests
import time
import base64
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load movie data and similarity matrix
movie_dict = pickle.load(open("movie.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))
movies = pd.DataFrame(movie_dict)

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    bg_image = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

set_background("image.jpeg")

API_KEY = "7d4965b57af7cbc033bdec119627b0b7"

# Fetch poster from TMDB
def fetch(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    retries = 3
    for _ in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
        except requests.exceptions.RequestException:
            time.sleep(2)
    return None

# 🔥 Fetch trending movies
def get_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/day?api_key={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        return [(movie['title'], "https://image.tmdb.org/t/p/w500" + movie['poster_path']) for movie in data['results'][:5]]
    except:
        return []

# Recommend by tags
def recommend_by_tags(query, top_n=5):
    filtered = movies[movies['tags'].str.contains(query.lower(), case=False, na=False)]
    return filtered['title'].head(top_n).tolist()

# Recommend by mood
def recommend_by_mood(query, top_n=5):
    sentiment = sia.polarity_scores(query)
    if sentiment['compound'] >= 0.05:
        filtered = movies[movies['tags'].str.contains('happy|comedy|fun|feel-good', case=False, na=False)]
    elif sentiment['compound'] <= -0.05:
        filtered = movies[movies['tags'].str.contains('dark|drama|sad|tragic', case=False, na=False)]
    else:
        filtered = movies.head(top_n)
    return filtered['title'].head(top_n).tolist()

# Display posters
def display_recommendations(titles, posters):
    cols = st.columns(min(5, len(titles)))
    for col, title, poster in zip(cols, titles, posters):
        with col:
            st.image(poster, caption=title, use_container_width=True)

# Streamlit UI
st.title("🎬 Movie Recommender System")

# 🔥 Trending movies section
st.subheader("🔥 Trending Movies Today")
trending = get_trending_movies()
if trending:
    t_cols = st.columns(len(trending))
    for col, (title, poster) in zip(t_cols, trending):
        with col:
            st.image(poster, caption=title, use_container_width=True)
else:
    st.write("Could not load trending movies at the moment.")

# Search bar (mic prompt removed)
st.subheader("🔍 Search")
query = st.text_input("Enter a movie title, actor, director, genre, or mood (e.g., happy, sad, etc.)")

mood_option = st.selectbox("Do you want mood-based recommendations?", ["No", "Yes"])

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        if mood_option == "Yes":
            titles = recommend_by_mood(query)
        else:
            titles = recommend_by_tags(query)

        posters = [fetch(movie_dict[movie_dict['title'] == title]['movie_id'].values[0]) for title in titles]
        display_recommendations(titles, posters)
