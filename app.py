import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
career_data = pd.read_csv('careers.csv')

# Preprocess skills for model training
career_data['skills_processed'] = career_data['skills'].apply(
    lambda x: ' '.join(x.replace("'", "").split(" "))
)


skills_with_urls = {
    'Python': 'https://www.freecodecamp.org/learn/scientific-computing-with-python/',
    'ML': 'https://www.kaggle.com/learn/machine-learning',
    'Data Analysis': 'https://www.datacamp.com/learn/data-analysis',
    'Statistics': 'https://www.khanacademy.org/math/statistics-probability',
    'AI': 'https://www.elementsofai.com/',
    'Deep Learning': 'https://www.deeplearning.ai/',
    'NLP': 'https://www.kaggle.com/learn/natural-language-processing',
    'HTML': 'https://www.freecodecamp.org/learn/responsive-web-design/',
    'CSS': 'https://www.freecodecamp.org/learn/responsive-web-design/',
    'JavaScript': 'https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/',
    'React': 'https://react.dev/learn',
    'Leadership': 'https://www.coursera.org/learn/leadership-skills-for-success',
    'Planning': 'https://www.mindtools.com/pages/article/newPPM_85.htm',
    'Communication': 'https://www.khanacademy.org/college-careers-more/career-content/home/skills-for-success-podcast/v/episode-4',
    'Java': 'https://www.codecademy.com/learn/learn-java',
    'C++': 'https://www.learncpp.com/',
    'Problem Solving': 'https://www.khanacademy.org/college-careers-more/soft-skills/creativity-problem-solving-and-innovation',
    'Creativity': 'https://www.coursera.org/learn/creativity-innovation-entrepreneurship',
    'Photoshop': 'https://helpx.adobe.com/photoshop/tutorials.html',
    'Illustrator': 'https://helpx.adobe.com/illustrator/tutorials.html',
    'UX': 'https://www.interaction-design.org/',
    'Excel': 'https://www.microsoft.com/en-us/training/excel',
    'Networking': 'https://www.cisco.com/c/en/us/training-events/training-certifications/exam-topics/ccna.html',
    'Security': 'https://www.cybrary.it/course/cyber-security/',
    'Risk Management': 'https://www.coursera.org/learn/enterprise-risk-management',
    'AWS': 'https://www.aws.training/',
    'Azure': 'https://learn.microsoft.com/en-us/training/browse/',
    'GCP': 'https://cloud.google.com/training',
    'CI/CD': 'https://www.redhat.com/en/topics/devops/what-is-ci-cd',
    'Docker': 'https://www.docker.com/101-tutorial/',
    'Kubernetes': 'https://kubernetes.io/docs/tutorials/',
    'Linux': 'https://ubuntu.com/tutorials',
    'CAD': 'https://www.autodesk.com/certification/overview',
    'SolidWorks': 'https://my.solidworks.com/training',
    'Thermodynamics': 'https://ocw.mit.edu/courses/mechanical-engineering/2-005-thermal-fluids-engineering-i-fall-2011/',
    'Material Science': 'https://ocw.mit.edu/courses/materials-science-and-engineering/',
    'AutoCAD': 'https://www.autodesk.com/education/free-software/autocad',
    'Structural Analysis': 'https://nptel.ac.in/courses/105/106/105106150/',
    'Construction': 'https://ocw.mit.edu/courses/civil-and-environmental-engineering/1-010-introduction-to-civil-engineering-design-spring-2010/',
    'SEO': 'https://moz.com/learn/seo',
    'Content Creation': 'https://www.hubspot.com/resources/courses',
    'Analytics': 'https://www.google.com/analytics/learn/',
    'Social Media': 'https://buffer.com/resources/social-media-strategy/',
    'Unity': 'https://learn.unity.com/',
    'Unreal Engine': 'https://www.unrealengine.com/en-US/onlinelearning',
    'C#': 'https://learn.microsoft.com/en-us/dotnet/csharp/',
    '3D Modeling': 'https://www.blender.org/support/tutorials/',
    'Android': 'https://developer.android.com/courses',
    'iOS': 'https://developer.apple.com/learn/',
    'React Native': 'https://reactnative.dev/docs/getting-started',
    'Flutter': 'https://docs.flutter.dev/get-started',
    'SQL': 'https://www.datacamp.com/courses/introduction-to-sql',
    'ETL': 'https://www.coursera.org/learn/data-pipelines-etl',
    'Big Data': 'https://www.edx.org/professional-certificate/big-data-for-data-engineering',
    'Hadoop': 'https://hadoop.apache.org/docs/',
    'TensorFlow': 'https://www.tensorflow.org/learn',
    'Math': 'https://www.youtube.com/@AlexMathsEngineering',
    'PyTorch': 'https://pytorch.org/tutorials/',
    'Recruitment': 'https://www.linkedin.com/learning/',
    'Training': 'https://www.trainup.com/',
    'Employee Engagement': 'https://www.coursera.org/learn/employee-engagement',
    'Conflict Resolution': 'https://www.khanacademy.org/college-careers-more/soft-skills/soft-skills-teamwork-leadership',
    'Writing': 'https://www.coursera.org/learn/the-strategy-of-content-marketing',
    'Editing': 'https://www.academiceditingservices.com/',
    'Research': 'https://www.coursera.org/learn/research-writing',
    'Adobe Premiere': 'https://helpx.adobe.com/premiere-pro/tutorials.html',
    'Final Cut Pro': 'https://support.apple.com/final-cut-pro',
    'Storyboarding': 'https://www.toonboom.com/resources',
    'Financial Analysis': 'https://corporatefinanceinstitute.com/resources/financial-analysis/',
}



# Initialize Flask app
app = Flask(__name__)

# Train TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(career_data['skills_processed'])

# Function to find missing skills (with exact skill matching)
def find_missing_skills(user_skills, career_skills):
    # Split user skills into list of skills
    user_skills_list = set(user_skills.split(', '))
    
    # Extract career's required skills as a set
    career_skills_list = career_skills.strip("'").split("' '")
    # print(career_skills_list)

    # Find missing skills (those in career's skills but not in user's skills)
    missing_skills = set(career_skills_list) - user_skills_list

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
        # print(career['skills'])
        missing_skills = find_missing_skills(user_skills, career['skills'])
        
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

    for i, recommendation in enumerate(recommendations):
        numerator = len(recommendation['missing_skills'])
        denominator = len(list(career_data[career_data['career'] == recommendation['career']]['skills'])[0].strip("'").split("' '"))
        recommendation['match_score'] = round(np.float64(((denominator-numerator)/denominator) * 100), 2)
    print(recommendations)

    return recommendations

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    user_skills = request.form['skills']
    recommendations = predict_career_match(user_skills)
    # print(recommendations)
    return render_template('result.html', recommendations=recommendations, skills_with_urls = skills_with_urls)

if __name__ == '__main__':
    app.run(debug=True)
