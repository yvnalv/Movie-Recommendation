import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda(df):
    # Distribution of content types
    sns.countplot(x='type', data=df)
    plt.title('Content Type Distribution')
    plt.savefig('reports/figures/content_type_distribution.png')
    
    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('reports/figures/correlation_heatmap.png')

if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_netflix_titles.csv')
    eda(df)
