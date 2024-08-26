from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the precomputed similarity matrix and the dataset
df = pd.read_csv('../data/raw/netflix_titles.csv')
cosine_sim = joblib.load('../models/similarity_matrix.pkl')

def get_recommendations(title, df, cosine_sim):
    idx = df[df['title'].str.lower() == title.lower()].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form['title']
    recommendations = get_recommendations(title, df, cosine_sim)
    return render_template('result.html', recommendations=recommendations, title=title)

if __name__ == "__main__":
    app.run(debug=True)
