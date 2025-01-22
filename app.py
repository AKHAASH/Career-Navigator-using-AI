from flask import Flask, request, render_template, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Load career data
career_data = pd.read_csv('careers.csv')

# Function to recommend careers based on user skills
def recommend_careers(user_skills):
    # Convert user skills to lowercase and split by comma
    user_skills_list = [skill.strip().lower() for skill in user_skills.split(",")]
    recommendations = []

    for _, row in career_data.iterrows():
        # Extract required skills and clean them (split by ' ', remove single quotes)
        required_skills = [skill.strip().replace("'", "").lower() for skill in row['skills'].split("' '")]
        
        # Calculate match score based on overlapping skills
        match_score = 0
        for user_skill in user_skills_list:
            if user_skill in required_skills:
                match_score += 1

        # Only recommend careers with at least one matching skill
        if match_score > 0:
            recommendations.append({
                "career": row['career'],
                "description": row['description'],
                "match_score": round((match_score / len(required_skills)) * 100, 2)
            })

    # Sort recommendations by match score (highest first)
    recommendations = sorted(recommendations, key=lambda x: x['match_score'], reverse=True)
    return recommendations

# Home route
@app.route('/')
def index():
    print(career_data.head())
    return render_template('index.html')

# Route to handle recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    user_skills = request.form.get('skills')
    if not user_skills:
        return render_template('index.html', error="Please enter your skills.")
    
    recommendations = recommend_careers(user_skills)
    return render_template('results.html', recommendations=recommendations, user_skills=user_skills)

if __name__ == '__main__':
    app.run(debug=True)
