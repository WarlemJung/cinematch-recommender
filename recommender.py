import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Global settings for Pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class MovieRecommenderSystem:
    def __init__(self, filepath, sample_size=20000):
        """
        Initializes the system by loading the data and applying basic preprocessing.
        :param filepath: Path to the CSV file.
        :param sample_size: Row limit to avoid memory overflow (RAM) when calculating the matrix.
        """
        print(f"--- Loading data from {filepath} ---")
        try:
            self.df = pd.read_csv(filepath, low_memory=False)
            
            # Filter essential columns
            self.df = self.df[['title', 'overview', 'vote_average', 'vote_count', 'release_date']]
            
            # Basic cleaning
            self.df['overview'] = self.df['overview'].fillna('')
            
            # Sampling for performance (optional, but recommended for local dev)
            self.df = self.df.iloc[0:sample_size].reset_index(drop=True)
            
            print(f"Data loaded: {self.df.shape[0]} movies.")
            
        except FileNotFoundError:
            raise Exception(f"File {filepath} not found. Check the path.")

    def get_demographic_recommendations(self, top_n=10):
        """
        Implements the IMDB 'Weighted Rating' formula.
        Returns the highest rated and most popular movies.
        """
        print("\n--- Calculating Demographic Recommendation (Top Charts) ---")
        
        # C: The mean vote across the whole dataset
        C = self.df['vote_average'].mean()
        
        # m: The minimum votes required to be listed (90% cutoff)
        m = self.df['vote_count'].quantile(0.9)
        
        # Filter movies that qualify for the chart
        q_movies = self.df.copy().loc[self.df['vote_count'] >= m]
        
        # Weighted rating function (IMDB Formula)
        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v/(v+m) * R) + (m/(v+m) * C)
        
        # Calculate score
        q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
        
        # Sort and return
        q_movies = q_movies.sort_values('score', ascending=False)
        return q_movies[['title', 'vote_count', 'vote_average', 'score']].head(top_n)

    def compute_content_similarity(self):
        """
        Prepares the TF-IDF matrix and calculates cosine similarity.
        Must be called before requesting content-based recommendations.
        """
        print("\n--- Processing TF-IDF Matrix and Cosine Similarity ---")
        
        # Remove common English stop words
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Construct TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(self.df['overview'])
        
        # Calculate similarity (Linear Kernel is faster than cosine_similarity for TF-IDF)
        self.cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Create reverse mapping of indices
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        print("Content model ready.")

    def get_content_recommendations(self, title, top_n=10):
        """
        Returns similar movies based on plot summary.
        """
        # Check if model has been trained
        if not hasattr(self, 'cosine_sim'):
            self.compute_content_similarity()

        try:
            # Get the index of the movie given the title
            idx = self.indices[title]
        except KeyError:
            return f"Error: The movie '{title}' was not found in the database."

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top_n most similar movies (excluding the first one, which is itself)
        sim_scores = sim_scores[1:top_n+1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]

        # Return titles
        return self.df['title'].iloc[movie_indices]

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Instantiate the system
    # Ensure movies_metadata.csv is in the same directory
    recsys = MovieRecommenderSystem('movies_metadata.csv')

    # 2. Test: Demographic Recommendation (General)
    top_movies = recsys.get_demographic_recommendations()
    print(top_movies)

    # 3. Prepare content model
    recsys.compute_content_similarity()

    # 4. Test: Content-Based Recommendation (Specific)
    movie_to_search = input("Enter movie name: ") 
    print(f"\n--- Movies similar to '{movie_to_search}' ---")
    content_recs = recsys.get_content_recommendations(movie_to_search)
    print(content_recs)