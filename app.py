import streamlit as st
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Title
st.title("Movie Recommendation System")

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("movies_metadata.csv", dtype={'id': str}, low_memory=False)
    df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != pd.NaT else None)
    
    # Fill missing values for 'tagline' and 'overview' columns
    df['tagline'] = df['tagline'].fillna('')
    df['overview'] = df['overview'].fillna('')
    
    # Create the 'description' column by combining 'overview' and 'tagline'
    df['description'] = df['overview'] + ' ' + df['tagline']
    df['description'] = df['description'].fillna('').str.strip()
    
    # Handle vote counts and averages
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    
    # Compute C and m values for weighted rating
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    
    return df, C, m

# Load the data
df, C, m = load_data()

# Filter out rows where the description is empty or contains only stop words
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
df = df.drop([19730, 29503, 35587])
df['id'] = df['id'].astype('int')
smd = df[df['id'].isin(links_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')




smd = smd.reset_index(drop=True) 
titles = smd['title']

valid_descriptions = smd[smd['description'].apply(lambda x: len(x.strip()) > 0)]

# Vectorization function to ensure valid descriptions
@st.cache_data
def vectorize_data(valid_descriptions):
    # Initialize TfidfVectorizer with stop words
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), stop_words='english', max_df=0.7, min_df=2)
    tfidf_matrix = tf.fit_transform(smd['description'])
  
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

# Apply vectorization on valid descriptions
cosine_sim = vectorize_data(smd)

# Recommendation function
def get_recommendations(title):
    title = title.lower()
    
    # Ensure title is in the valid descriptions
    indices = pd.Series(smd.index, index=smd['title'].str.lower())
    if title not in indices:
        raise KeyError(f"{title} not found in dataset")
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
    
    

# Streamlit UI for recommendations
movie_name = st.text_input("Enter the name of a movie")
if st.button("Recommend Movies"):
    try:
        recommendations = get_recommendations(movie_name)
        st.write("Top movie recommendations:")
        st.write(recommendations.head(10))
    except KeyError:
        st.write("Movie not found in the dataset. Please try another title.")
