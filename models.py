import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.build_model()

    def build_model(self):
        self.movies_df['Description'] = self.movies_df['Description'].fillna('')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf_vectorizer.fit_transform(self.movies_df['Description'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, movie_title):
        movie_title_lower = movie_title.lower()  # Convert input to lowercase for case-insensitive matching
        idx_list = []
        for idx, title in enumerate(self.movies_df['Title']):
            if movie_title_lower in title.lower():  # Case-insensitive search
                idx_list.append(idx)

        if not idx_list:
            return []  # Return an empty list if no movies are found

        sim_scores = []
        for idx in idx_list:
            sim_scores.extend(list(enumerate(self.cosine_sim[idx])))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Get top 10 similar movies
        movie_indices = [score[0] for score in sim_scores]
        recommendations = self.movies_df['Title'].iloc[movie_indices].tolist()
        return recommendations
