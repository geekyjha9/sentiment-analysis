
from flask import Flask, request, jsonify
import joblib
from textblob import TextBlob
import numpy as np

app = Flask(__name__)

# Load the saved model
loaded_model = joblib.load('./sentiment_model.pkl')

# Function to calculate sentiment polarity and normalize sentiment score
def calculate_rating(comments):
    polarity_scores = [TextBlob(comment).sentiment.polarity for comment in comments]
    normalized_polarity_scores = [(score + 1) * 2.5 for score in polarity_scores]
    overall_sentiment = np.mean(normalized_polarity_scores)
    return overall_sentiment

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'comments' not in data or not isinstance(data['comments'], list):
        return jsonify({'error': 'Invalid input format. Expected JSON object with "comments" as a list of strings.'}), 400
    
    comments = data['comments']
    rating = calculate_rating(comments)
    rating_prediction = loaded_model.predict([[rating]])
    rating_prediction_clipped = np.clip(rating_prediction, 0, 5)
    
    return jsonify({'predicted_rating': float(rating_prediction_clipped[0])})

if __name__ == '__main__':
    app.run(debug=True)
