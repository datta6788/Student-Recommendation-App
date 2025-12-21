from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
import secrets
print(secrets.token_hex(32))
warnings.filterwarnings('ignore')
from datetime import timedelta
app = Flask(__name__)
app.permanent_session_lifetime = timedelta(minutes=30)
app = Flask(__name__)
app.secret_key = 'your_very_secure_random_key_here'
app = Flask(__name__)
from werkzeug.security import generate_password_hash, check_password_hash
   
   # Store hashed passwords
hashed = generate_password_hash('password123')
   
   # Verify passwords
check_password_hash(hashed, 'password123')

USERS = {
    'admin': {'password': 'admin123', 'role': 'Administrator'},
    'teacher': {'password': 'teacher123', 'role': 'Teacher'},
    'newuser': {'password': 'newpass', 'role': 'Custom Role'}
}
# Your CSV data
csv_data = """student_id,name,maths,science,english,attendance,interest,result
1,Aarav Sharma,78,82,75,75,AI,0
2,Ananya Reddy,65,70,68,72,Web Development,1
3,Rohan Verma,90,88,92,95,Data Science,0
4,Sneha Patel,55,60,58,65,Cybersecurity,0
5,Karthik Kumar,72,75,70,80,AI,0
6,Priya Singh,70,75,50,60,Web Development,0
7,Aditya Mehta,85,80,78,90,Data Science,0
8,Neha Gupta,62,65,60,70,Cybersecurity,1
9,Vikram Rao,88,90,85,93,AI,0
10,Pooja Iyer,45,50,48,58,Web Development,1
11,Siddharth Malhotra,70,72,68,78,Data Science,0
12,Kavya Nair,82,85,80,88,AI,0
13,Rahul Choudhary,60,62,65,70,Cybersecurity,1
14,Divya Jain,92,90,94,96,Data Science,0
15,Arjun Das,55,58,60,68,Web Development,1
16,Meghana Kulkarni,75,78,72,85,AI,0
17,Amit Yadav,68,70,65,75,Data Science,0
18,Swathi Gopal,50,55,52,60,Cybersecurity,1
19,Nikhil Bansal,88,85,90,92,AI,0
20,Shreya Agarwal,42,45,48,55,Web Development,1
21,Manish Pandey,78,80,76,86,Data Science,0
22,Aishwarya Mishra,65,68,70,74,Cybersecurity,1
23,Sandeep Khatri,90,92,88,95,AI,0
24,Ritu Saxena,58,60,62,68,Web Development,1
25,Varun Kapoor,72,75,78,82,Data Science,0
26,Bhavya Shah,48,50,52,60,Cybersecurity,1
27,Pranav Joshi,85,88,84,90,AI,0
28,Keerthi Ramesh,60,62,65,70,Web Development,1
29,Akash Tripathi,92,94,90,96,Data Science,0
30,Nandini Roy,55,58,60,65,Cybersecurity,1
31,Harsh Vardhan,75,78,72,85,AI,0
32,Tanvi Deshpande,68,70,65,75,Web Development,0
33,Mohit Arora,80,82,78,88,Data Science,0
34,Ishita Sen,50,55,52,60,Cybersecurity,1
35,Rakesh Pillai,88,85,90,92,AI,0
36,Anjali Menon,45,48,50,55,Web Development,1
37,Deepak Soni,78,80,76,86,Data Science,0
38,Pallavi Kulkarni,62,65,60,70,Cybersecurity,1
39,Abhishek Tiwari,90,92,88,95,AI,0
40,Sumanth Reddy,55,58,60,68,Web Development,1
41,Kunal Khanna,72,75,78,82,Data Science,0
42,Lakshmi Narayanan,48,50,52,60,Cybersecurity,1
43,Yash Agarwal,85,88,84,90,AI,0
44,Chaitanya Kulkarni,60,62,65,70,Web Development,1
45,Ravi Teja,92,94,90,96,Data Science,0
46,Monika Bhat,55,58,60,65,Cybersecurity,1
47,Saurabh Srivastava,75,78,72,85,AI,0
48,Aparna Rao,68,70,65,75,Web Development,0
49,Ritesh Mahajan,80,82,78,88,Data Science,0
50,Sindhu Lakshmi,50,55,52,60,Cybersecurity,1
51,Naveen Kumar,88,85,90,92,AI,0
52,Pankaj Singh,45,48,50,55,Web Development,1
53,Mehul Patel,78,80,76,86,Data Science,0
54,Shalini Verma,62,65,60,70,Cybersecurity,1
55,Ashwin Iyer,90,92,88,95,AI,0
56,Kriti Malhotra,55,58,60,68,Web Development,1
57,Uday Kiran,72,75,78,82,Data Science,0
58,Bhavana Shetty,48,50,52,60,Cybersecurity,1
59,Rahul Mehra,85,88,84,90,AI,0
60,Ankita Joshi,60,62,65,70,Web Development,1"""

# Load data
df = pd.read_csv(StringIO(csv_data))

# Machine Learning Models
class StudentRiskMLModel:
    def __init__(self):
        self.risk_classifier = None
        self.performance_predictor = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def prepare_features(self, data):
        """Prepare features for ML model"""
        features = pd.DataFrame()
        features['avg_marks'] = (data['maths'] + data['science'] + data['english']) / 3
        features['attendance'] = data['attendance']
        features['maths'] = data['maths']
        features['science'] = data['science']
        features['english'] = data['english']
        features['marks_std'] = data[['maths', 'science', 'english']].std(axis=1)
        features['attendance_marks_ratio'] = data['attendance'] / (features['avg_marks'] + 1)
        features['lowest_subject'] = data[['maths', 'science', 'english']].min(axis=1)
        
        return features
    
    def train_models(self, data):
        """Train ML models on student data"""
        # Prepare features
        X = self.prepare_features(data)
        
        # Create risk labels based on attendance and marks
        data['risk_label'] = ((data['attendance'] < 75) | 
                             ((data['maths'] + data['science'] + data['english'])/3 < 65)).astype(int)
        
        y_risk = data['risk_label']
        y_performance = (data['maths'] + data['science'] + data['english']) / 3
        
        # Train Risk Classification Model
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.risk_classifier.fit(X_scaled, y_risk)
        
        # Train Performance Prediction Model
        self.performance_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.performance_predictor.fit(X_scaled, y_performance)
        
        # Store feature importance
        feature_names = X.columns.tolist()
        self.feature_importance = dict(zip(
            feature_names, 
            self.risk_classifier.feature_importances_
        ))
        
        return {
            'risk_accuracy': self.risk_classifier.score(X_scaled, y_risk),
            'performance_r2': self.performance_predictor.score(X_scaled, y_performance)
        }
    
    def predict_risk(self, student_features):
        """Predict risk probability using ML"""
        X_scaled = self.scaler.transform(student_features)
        risk_proba = self.risk_classifier.predict_proba(X_scaled)[0]
        predicted_performance = self.performance_predictor.predict(X_scaled)[0]
        
        return {
            'risk_probability': float(risk_proba[1] * 100),  # Probability of being at risk
            'predicted_performance': float(predicted_performance),
            'confidence': float(max(risk_proba) * 100)
        }

# Initialize and train ML model
ml_model = StudentRiskMLModel()
training_metrics = ml_model.train_models(df)

def calculate_ml_risk_analysis(student_data):
    """Calculate risk using ML model"""
    # Prepare student features
    student_df = pd.DataFrame([student_data])
    features = ml_model.prepare_features(student_df)
    
    # Get ML predictions
    ml_prediction = ml_model.predict_risk(features)
    
    avg_marks = (student_data['maths'] + student_data['science'] + student_data['english']) / 3
    attendance = student_data['attendance']
    
    # Identify risk factors with ML insights
    risk_factors = []
    
    # Attendance analysis
    if attendance < 75:
        risk_factors.append({
            'factor': f"Critical: Attendance at {attendance}% (Below 75% threshold)",
            'impact': 'High',
            'ml_weight': ml_model.feature_importance.get('attendance', 0) * 100
        })
    elif attendance < 85:
        risk_factors.append({
            'factor': f"Moderate: Attendance at {attendance}% (Should be above 85%)",
            'impact': 'Medium',
            'ml_weight': ml_model.feature_importance.get('attendance', 0) * 100
        })
    
    # Performance analysis
    if avg_marks < 60:
        risk_factors.append({
            'factor': f"Critical: Average marks {avg_marks:.1f}% (Below passing threshold)",
            'impact': 'High',
            'ml_weight': ml_model.feature_importance.get('avg_marks', 0) * 100
        })
    elif avg_marks < 70:
        risk_factors.append({
            'factor': f"Warning: Average marks {avg_marks:.1f}% (Below expected standard)",
            'impact': 'Medium',
            'ml_weight': ml_model.feature_importance.get('avg_marks', 0) * 100
        })
    
    # Subject-specific analysis
    weak_subjects = []
    if student_data['maths'] < 60:
        weak_subjects.append('Mathematics')
    if student_data['science'] < 60:
        weak_subjects.append('Science')
    if student_data['english'] < 60:
        weak_subjects.append('English')
    
    if weak_subjects:
        risk_factors.append({
            'factor': f"Weak subjects identified: {', '.join(weak_subjects)}",
            'impact': 'High',
            'ml_weight': ml_model.feature_importance.get('lowest_subject', 0) * 100
        })
    
    # Performance consistency
    marks_std = np.std([student_data['maths'], student_data['science'], student_data['english']])
    if marks_std > 15:
        risk_factors.append({
            'factor': f"Inconsistent performance across subjects (variance: {marks_std:.1f})",
            'impact': 'Medium',
            'ml_weight': ml_model.feature_importance.get('marks_std', 0) * 100
        })
    
    # Previous result
    if student_data['result'] == 1:
        risk_factors.append({
            'factor': "Previous academic challenges on record",
            'impact': 'Medium',
            'ml_weight': 15.0
        })
    
    return {
        'ml_risk_probability': ml_prediction['risk_probability'],
        'predicted_future_performance': ml_prediction['predicted_performance'],
        'confidence_score': ml_prediction['confidence'],
        'avg_marks': round(avg_marks, 2),
        'is_at_risk': ml_prediction['risk_probability'] > 50,
        'risk_factors': risk_factors,
        'ml_insights': {
            'top_risk_factor': max(ml_model.feature_importance.items(), key=lambda x: x[1])[0],
            'feature_importance': ml_model.feature_importance
        }
    }

def get_ml_improvement_suggestions(student_data, risk_analysis):
    """Generate ML-based personalized improvement suggestions"""
    suggestions = []
    
    predicted_improvement = risk_analysis['predicted_future_performance'] - risk_analysis['avg_marks']
    
    # Priority-based suggestions using ML insights
    top_factors = sorted(
        risk_analysis['risk_factors'], 
        key=lambda x: x['ml_weight'], 
        reverse=True
    )
    
    if student_data['attendance'] < 75:
        suggestions.append({
            'category': '🎯 Priority #1: Attendance Recovery',
            'priority': 'CRITICAL',
            'ml_impact': f"{ml_model.feature_importance.get('attendance', 0)*100:.1f}% impact on success",
            'points': [
                f"ML Analysis: Attendance is the #{1} predictor of your risk level",
                'Target: Achieve 85%+ attendance in next 30 days',
                'Action: Set up automated attendance reminders',
                'Meet with counselor to address attendance barriers',
                'Track daily attendance progress using mobile app'
            ]
        })
    
    if risk_analysis['avg_marks'] < 70:
        improvement_potential = risk_analysis['predicted_future_performance'] - risk_analysis['avg_marks']
        suggestions.append({
            'category': '📚 Priority #2: Academic Performance Boost',
            'priority': 'HIGH',
            'ml_impact': f"Potential to improve by {abs(improvement_potential):.1f} marks",
            'points': [
                f"ML Prediction: You can reach {risk_analysis['predicted_future_performance']:.1f}% with consistent effort",
                'Create personalized study plan: 2-3 hours daily',
                'Focus on concept understanding, not memorization',
                'Join peer study groups (proven to boost performance by 15-20%)',
                'Schedule weekly review sessions for all subjects'
            ]
        })
    
    # Subject-specific ML recommendations
    subject_scores = {
        'Mathematics': student_data['maths'],
        'Science': student_data['science'],
        'English': student_data['english']
    }
    
    weakest_subject = min(subject_scores.items(), key=lambda x: x[1])
    
    if weakest_subject[1] < 65:
        suggestions.append({
            'category': f'🎓 Focus Area: {weakest_subject[0]} Improvement',
            'priority': 'HIGH',
            'ml_impact': 'Weakest subject - high improvement potential',
            'points': [
                f"Current: {weakest_subject[1]}% → Target: 75%+ (10+ marks boost)",
                'Daily practice: 30 minutes minimum',
                'Use Khan Academy/YouTube for concept clarification',
                'Solve previous year papers (focus on common patterns)',
                'Get peer tutoring or teacher consultation twice weekly'
            ]
        })
    
    # Smart study techniques based on ML analysis
    suggestions.append({
        'category': '🧠 ML-Recommended Study Strategies',
        'priority': 'MEDIUM',
        'ml_impact': 'Optimized based on successful student patterns',
        'points': [
            'Pomodoro Technique: 25 min study + 5 min break (proven effective)',
            'Active recall: Test yourself instead of re-reading',
            'Spaced repetition: Review topics at increasing intervals',
            'Teach concepts to others (improves retention by 90%)',
            f'Leverage your interest in {student_data["interest"]} for motivated learning'
        ]
    })
    
    # Personalized motivation
    if risk_analysis['is_at_risk']:
        suggestions.append({
            'category': '💪 Motivation & Mindset',
            'priority': 'MEDIUM',
            'ml_impact': 'Essential for sustained improvement',
            'points': [
                'Set micro-goals: Weekly targets instead of semester goals',
                'Celebrate small wins: Track daily progress',
                'Find an accountability partner or study buddy',
                'Visualize success: You have the potential to improve',
                'Remember: Current performance does not define future potential'
            ]
        })
    
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/students')
def get_students():
    """Get list of all students"""
    students = df[['student_id', 'name']].to_dict('records')
    return jsonify(students)

@app.route('/api/student/<int:student_id>')
def get_student_report(student_id):
    """Get complete ML-powered report for a student"""
    student = df[df['student_id'] == student_id].iloc[0]
    
    risk_analysis = calculate_ml_risk_analysis(student)
    suggestions = get_ml_improvement_suggestions(student, risk_analysis)
    
    report = {
        'student_id': int(student['student_id']),
        'name': student['name'],
        'attendance': float(student['attendance']),
        'marks': {
            'maths': int(student['maths']),
            'science': int(student['science']),
            'english': int(student['english'])
        },
        'interest': student['interest'],
        'avg_marks': risk_analysis['avg_marks'],
        'ml_risk_probability': round(risk_analysis['ml_risk_probability'], 2),
        'predicted_future_performance': round(risk_analysis['predicted_future_performance'], 2),
        'confidence_score': round(risk_analysis['confidence_score'], 2),
        'is_at_risk': risk_analysis['is_at_risk'],
        'risk_factors': risk_analysis['risk_factors'],
        'suggestions': suggestions,
        'ml_insights': risk_analysis['ml_insights']
    }
    
    return jsonify(report)

@app.route('/api/ml-metrics')
def get_ml_metrics():
    """Get ML model performance metrics"""
    return jsonify(training_metrics)

if __name__ == '__main__':
    print("=" * 60)
    print("ML Model Training Complete!")
    print(f"Risk Prediction Accuracy: {training_metrics['risk_accuracy']*100:.2f}%")
    print(f"Performance Prediction R²: {training_metrics['performance_r2']:.4f}")
    print("=" * 60)
    app.run(debug=True)
