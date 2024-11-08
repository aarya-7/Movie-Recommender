import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from ast import literal_eval
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Title for the app
st.title("Movie Recommendation System")

# Load datasets
@st.cache_data
def load_data():
    df = pd.read_csv("movies_metadata.csv")
    df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    df['id'] = df['id'].astype('int', errors='ignore')
    return df

df = load_data()

@st.cache_data
def load_links():
    links_small = pd.read_csv('links_small.csv')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    return links_small

links_small = load_links()

# Data preprocessing
df = df.drop([19730, 29503, 35587])
df = df[df['id'].isin(links_small)]

# Vectorizing
@st.cache_data
def vectorize_data():
    df['description'] = df['description'].fillna('')
    # Filter out rows with empty descriptions
    non_empty_descriptions = df[df['description'].str.strip() != '']
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
    tfidf_matrix = tf.fit_transform(non_empty_descriptions['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = vectorize_data()

# Indexing titles
smd = df.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Function for recommendations
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

# Input from user
movie_input = st.text_input("Enter a movie name", "Iron Man")

# Show recommendations
if st.button("Recommend Movies"):
    try:
        recommendations = get_recommendations(movie_input)
        st.write("Top movie recommendations:")
        st.write(recommendations.head(10))
    except KeyError:
        st.write("Movie not found in the dataset. Please try another title.")

# Hybrid Recommendation Functionality (Optional if using SVD and ratings)
ratings = pd.read_csv('ratings_small.csv')
reader = Reader()
svd = SVD()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
svd.fit(trainset)

# Function to recommend movies using SVD (collaborative filtering)
def hybrid(userId, title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, x).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)

# Optional: Include user-based filtering
user_id = st.number_input("Enter user ID for collaborative filtering", min_value=1, step=1)
if st.button("Hybrid Recommendations"):
    try:
        recommendations = hybrid(user_id, movie_input)
        st.write("Top hybrid recommendations:")
        st.write(recommendations)
    except KeyError:
        st.write("Movie not found in the dataset.")
