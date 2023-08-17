from flask import Flask, render_template, request
from models import MovieRecommender
import utils

app = Flask(__name__)

movies_df = utils.load_movies_data('movies.csv')
movie_recommender = MovieRecommender(movies_df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    recommendations = movie_recommender.get_recommendations(user_input)
    
    if not recommendations:
        message = f"No recommendations found for '{user_input}'."
        return render_template('index.html', message=message)

    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
