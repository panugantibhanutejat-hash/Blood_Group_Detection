from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
from calibrated_predictor import predict_blood_group_calibrated as predict_blood_group
from pdf_report import create_blood_group_report

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Blood group labels
blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    predicted_blood_group = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('detections', lazy=True))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create tables
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def home():
    # Get statistics for the portfolio page
    total_users = User.query.count()
    total_detections = Detection.query.count()
    return render_template('home.html', total_users=total_users, total_detections=total_detections)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Find user in database
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['logged_in'] = True
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('detector'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('signup.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/detector')
def detector():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    # Get user's detection history
    detections = Detection.query.filter_by(user_id=session['user_id']).order_by(Detection.created_at.desc()).limit(5).all()
    return render_template('detector.html', detections=detections)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("=== PREDICT ROUTE CALLED ===")
        
        if not session.get('logged_in'):
            print("User not logged in")
            return jsonify({'error': 'Unauthorized'}), 401
            
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            print("File validation passed")
            # Generate unique filename
            filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            print("File saved successfully")
            
            try:
                print("Calling predictor function...")
                # Use your existing ResNet model with enhanced preprocessing
                result = predict_blood_group(filepath)
                print(f"Predictor returned: {result}")
                
                # Handle different return formats
                if len(result) == 3:
                    predicted_blood_group, confidence, top_3_predictions = result
                elif len(result) >= 4:
                    predicted_blood_group, confidence, top_3_predictions, _ = result[:4]
                else:
                    predicted_blood_group, confidence, top_3_predictions = None, 0.0, []
                
                # Convert top_3_predictions format if needed
                if top_3_predictions and 'calibrated_confidence' in top_3_predictions[0]:
                    # Convert calibrated format to standard format
                    converted_predictions = []
                    for pred in top_3_predictions:
                        converted_predictions.append({
                            'blood_group': pred['blood_group'],
                            'confidence': pred.get('calibrated_confidence', pred.get('confidence', 0.0))
                        })
                    top_3_predictions = converted_predictions
                
                if predicted_blood_group:
                    # Save detection to database
                    detection = Detection(
                        user_id=session['user_id'],
                        image_path=filepath,
                        predicted_blood_group=predicted_blood_group,
                        confidence=confidence
                    )
                    db.session.add(detection)
                    db.session.commit()
                    
                    response_data = {
                        'success': True,
                        'prediction': predicted_blood_group,
                        'confidence': confidence,
                        'top_3_predictions': top_3_predictions,
                        'image_url': f'/static/uploads/{filename}'
                    }
                    print(f"Sending response: {response_data}")
                    return jsonify(response_data)
                else:
                    print("Prediction failed or low confidence")
                    return jsonify({'error': 'Prediction failed or low confidence'}), 500
                    
            except Exception as e:
                print(f"Exception in prediction: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
                
        else:
            print("Invalid file type")
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG images.'}), 400
            
    except Exception as e:
        print(f"Unexpected error in predict route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Generate report content
        report_content = {
            'title': 'Blood Group Detection Report',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'prediction': data.get('prediction', 'N/A'),
            'confidence': f"{data.get('confidence', 0) * 100:.2f}%",
            'top_3_predictions': data.get('top_3_predictions', []),
            'image_url': data.get('image_url', ''),
            'user': session.get('username', 'Unknown')
        }
        
        return jsonify(report_content)
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
            
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Prepare report data
        report_data = {
            'prediction': data.get('prediction', 'N/A'),
            'confidence': f"{data.get('confidence', 0) * 100:.2f}%",
            'top_3_predictions': data.get('top_3_predictions', []),
            'image_url': data.get('image_url', ''),
            'user': session.get('username', 'Unknown')
        }
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"blood_group_report_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        
        # Create PDF report
        if create_blood_group_report(report_data, pdf_path):
            # Return download link
            return jsonify({
                'success': True,
                'download_url': f'/static/uploads/{pdf_filename}'
            })
        else:
            return jsonify({'error': 'Failed to generate report'}), 500
            
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)