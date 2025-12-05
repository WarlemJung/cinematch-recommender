from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from recommender import MovieRecommenderSystem
import os

app = Flask(__name__)
CORS(app)

# Initialize the Recommender System
# Ensure movies_metadata.csv is in the same directory
csv_path = os.path.join(os.path.dirname(__file__), 'movies_metadata.csv')
recsys = None

try:
    recsys = MovieRecommenderSystem(csv_path)
    # Pre-compute similarity matrix for faster runtime
    recsys.compute_content_similarity()
    print("System initialized successfully.")
except Exception as e:
    print(f"Error initializing system: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trending', methods=['GET'])
def get_trending():
    if not recsys:
        return jsonify({'error': 'System not initialized'}), 500
    
    try:
        # Get top 12 for the grid
        trending = recsys.get_demographic_recommendations(top_n=12)
        return jsonify(trending.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    if not recsys:
        return jsonify({'error': 'System not initialized'}), 500
    
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Title parameter is required'}), 400

    try:
        recommendations = recsys.get_content_recommendations(title, top_n=12)
        
        if isinstance(recommendations, str): # Error message from recommender
             return jsonify({'error': recommendations}), 404
             
        # Convert Series to list
        return jsonify(recommendations.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
