import pandas as pd

def load_and_clean_data():
    df = pd.read_csv('data/raw/netflix_titles.csv')
    
    # Fill missing values with "Unknown"
    df.fillna({'director': 'Unknown', 'cast': 'Unknown'}, inplace=True)
    
    # Drop rows with missing 'title' or 'listed_in' (genres)
    df.dropna(subset=['title', 'listed_in'], inplace=True)
    
    # Save cleaned data
    df.to_csv('data/processed/cleaned_netflix_titles.csv', index=False)
    return df

if __name__ == "__main__":
    load_and_clean_data()
