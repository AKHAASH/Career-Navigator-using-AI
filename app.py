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

# Function to find missing skills (with exact skill matching)
def find_missing_skills(user_skills, career_skills):
    # Split user skills into list of skills
    user_skills_list = set(user_skills.split(','))
    
    # Extract career's required skills as a set
    career_skills_list = set(career_skills.split(' '))

    # Find missing skills (those in career's skills but not in user's skills)
    missing_skills = career_skills_list - user_skills_list

    # Return missing skills as a list
    return list(missing_skills)

# Function to predict match percentage
def predict_career_match(user_skills):
    user_skills_processed = ' '.join(user_skills.replace("'", "").split(" "))
    user_vector = vectorizer.transform([user_skills_processed])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

    recommendations = []
    for i, score in enumerate(similarity_scores):
        # Skip careers with 0% match
        if score == 0:
            continue
        
        career = career_data.iloc[i]
        missing_skills = find_missing_skills(user_skills, career['skills_processed'])
        
        # Avoid displaying missing skills if the match is 100%
        if score == 1.0:
            missing_skills = set()  # No missing skills for 100% match
            
        recommendations.append({
            "career": career['career'],
            "description": career['description'],
            "match_score": round(score * 100, 2),  # Convert to percentage
            "missing_skills": list(missing_skills)  # Ensure it is a list
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
