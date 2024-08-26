import pandas as pd

def create_features(df):
    # Combine relevant metadata into a single feature
    df['metadata'] = df['director'] + ' ' + df['cast'] + ' ' + df['listed_in'] + ' ' + df['description']
    df.to_csv('data/processed/featured_netflix_titles.csv', index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_netflix_titles.csv')
    create_features(df)
