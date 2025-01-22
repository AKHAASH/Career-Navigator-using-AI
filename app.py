import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
career_data = pd.read_csv('careers.csv')

# Preprocess skills for model training
career_data['skills_processed'] = career_data['skills'].apply(
    lambda x: ' '.join(x.replace("'", "").split(" "))
)

# Initialize Flask app
app = Flask(__name__)

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(career_data['skills_processed'])

# Function to predict match percentage
def predict_career_match(user_skills):
    user_skills_processed = ' '.join(user_skills.replace("'", "").split(" "))
    user_vector = vectorizer.transform([user_skills_processed])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

    recommendations = []
    for i, score in enumerate(similarity_scores):
        if score > 0:  # Only include careers with some match
            recommendations.append({
                "career": career_data.iloc[i]['career'],
                "description": career_data.iloc[i]['description'],
                "match_score": round(score * 100, 2)  # Convert to percentage
            })

    # Sort by match score (highest first)
    recommendations = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)
    return recommendations

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_skills = request.form['skills']
    recommendations = predict_career_match(user_skills)
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
