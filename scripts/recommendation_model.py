import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def build_similarity_matrix():
    # Load the processed data
    df = pd.read_csv('data/processed/featured_netflix_titles.csv')
    
    # Use TF-IDF to vectorize the metadata
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['metadata'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Save the similarity matrix
    joblib.dump(cosine_sim, 'models/similarity_matrix.pkl')
    return df, cosine_sim

def get_recommendations(title, df, cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    
    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top 10 similar movies (excluding the input itself)
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 similar movies
    return df['title'].iloc[movie_indices].values

if __name__ == "__main__":
    df, cosine_sim = build_similarity_matrix()
    recommendations = get_recommendations('Breaking Bad', df, cosine_sim)
    print(recommendations)
