"""
=============================================================
FILE: app.py
STEP 3 OF THE AI PIPELINE — THE WEB SERVER

PURPOSE:
    This is the main file that runs the website.
    It loads the TRAINED AI model (from model_training.py)
    and uses it to analyse job postings submitted by users.

SIMPLE ANALOGY:
    model_training.py = Training a doctor (takes time, done once)
    app.py            = The doctor seeing patients every day (instant)
    
    The doctor (model) is trained ONCE.
    Then they use their knowledge to diagnose patients quickly.

HOW IT WORKS:
    1. When you run "python app.py", Flask starts a web server
    2. When you visit http://localhost:5000, you see the homepage
    3. When you paste a job and click Analyse:
       a. Flask receives your text (via HTTP POST)
       b. The text is cleaned using preprocess.py
       c. The cleaned text is converted to numbers (tokenizer)
       d. The numbers are fed into the Bi-LSTM model
       e. The model outputs a fraud probability (0.0 to 1.0)
       f. Flask sends the result back to your browser as JSON
       g. JavaScript displays the result without refreshing the page
=============================================================
"""

# ---- FLASK WEB FRAMEWORK ----
from flask import Flask, render_template, request, jsonify

# ---- STANDARD LIBRARY ----
import os
import json
import pickle

# ---- DATA SCIENCE ----
import numpy as np

# ---- TENSORFLOW ----
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- OUR PREPROCESSOR ----
from preprocess import TextPreprocessor

# ---- SUPPRESS TF LOGS (cleaner terminal output) ----
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# INITIALISE THE FLASK APPLICATION
# ============================================================
app = Flask(__name__)

# ============================================================
# LOAD THE TRAINED MODEL AND TOKENIZER AT STARTUP
# ============================================================
# These are loaded once when the server starts.
# This means prediction is INSTANT for every user request.

print("\n" + "="*60)
print("  FAKE JOB POSTING DETECTION SYSTEM — WEB SERVER")
print("  Olagunju Basheer Olaniyi | 220303010022 | LASUSTECH 2026")
print("="*60)

# Load model configuration
CONFIG_PATH = 'models/model_config.json'
MODEL_PATH = 'models/fake_job_model.keras'
TOKENIZER_PATH = 'models/tokenizer.pkl'

model = None
tokenizer = None
config = {}
model_loaded = False

if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    try:
        print("Loading trained AI model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        print("Loading tokenizer (word vocabulary)...")
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded successfully")
        
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        
        MAX_SEQUENCE_LENGTH = config.get('max_sequence_length', 300)
        model_loaded = True
        print("✅ System ready for predictions!\n")
        
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("⚠️ Running in DEMO MODE (no model loaded)")
        model_loaded = False
else:
    print("⚠️ No trained model found.")
    print("⚠️ Please run: python model_training.py")
    print("⚠️ Running in DEMO MODE.\n")
    MAX_SEQUENCE_LENGTH = 300
    model_loaded = False

# Initialise preprocessor (always available)
preprocessor = TextPreprocessor()

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_fraud(job_title, job_description, company_profile, requirements, benefits):
    """
    THE MAIN PREDICTION FUNCTION
    
    Takes raw job posting text as input.
    Returns a fraud probability (0.0 to 1.0).
    
    Processing pipeline:
    1. Combine all fields into one text block
    2. Clean and preprocess the text (our TextPreprocessor)
    3. Convert words to number sequences (tokenizer)
    4. Pad to fixed length (300 words)
    5. Feed to Bi-LSTM model
    6. Return prediction probability
    
    Example:
        Input:  title="URGENT!!! Work From Home", description="Earn $5000..."
        Output: 0.87  (87% likely to be fraud)
    """
    
    # Step 1: Combine all fields
    row = {
        'title': job_title,
        'description': job_description,
        'company_profile': company_profile,
        'requirements': requirements,
        'benefits': benefits
    }
    
    # Step 2: Preprocess text (clean, tokenize, lemmatize)
    cleaned_text = preprocessor.combine_fields(row)
    
    if not cleaned_text:
        return 0.0, "No text provided"
    
    # Step 3: Convert text to number sequence using saved tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    
    # Step 4: Pad to fixed length (same length as training)
    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding='pre',
        truncating='post'
    )
    
    # Step 5: Make prediction
    # model.predict returns shape (1, 1) — one sample, one output
    prediction = model.predict(padded, verbose=0)
    
    # Extract the scalar probability value
    probability = float(prediction[0][0])
    
    return probability, cleaned_text


def demo_predict(job_title, job_description, company_profile, requirements, benefits):
    """
    DEMO MODE PREDICTION (used when model is not trained yet)
    
    Uses the rule-based approach as a fallback.
    This is for demonstration/testing when the AI model has not been trained.
    Once you train the model (python model_training.py), this is replaced
    by the real Bi-LSTM neural network prediction.
    """
    import re, math
    
    all_text = " ".join([job_title, job_description, company_profile, requirements, benefits]).lower()
    
    fraud_signals = [
        (r'registration fee|application fee|processing fee|pay.*apply|payment required', 0.4),
        (r'bank account|bvn|nin|bank details|credit card number', 0.35),
        (r'earn.*week|guaranteed income|unlimited earning|passive income|easy money', 0.25),
        (r'gmail\.com|yahoo\.com|hotmail\.com|whatsapp|telegram', 0.2),
        (r'no experience required|no qualification|no degree needed', 0.15),
        (r'urgent|immediately|asap|apply now|limited slots', 0.1),
        (r'congratulations.*selected|you have been chosen|exclusive opportunity', 0.2),
        (r'work from anywhere|worldwide|no office|any country', 0.1),
    ]
    
    score = 0.0
    for pattern, weight in fraud_signals:
        if re.search(pattern, all_text):
            score += weight
    
    probability = 1 / (1 + math.exp(-8 * (score - 0.3)))
    return max(0.01, min(0.99, probability)), "demo_mode"


# ============================================================
# WEB ROUTES
# ============================================================

@app.route('/')
def index():
    """
    HOMEPAGE
    Renders the main form where users paste job postings.
    """
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/analyse', methods=['POST'])
def analyse():
    """
    ANALYSIS ENDPOINT
    
    Called when the user clicks "Analyse".
    
    Flow:
    1. Receive form data (job title, description, etc.)
    2. Validate input
    3. Run prediction (real model OR demo mode)
    4. Format and return results as JSON
    """
    try:
        # Get submitted form data
        job_title = request.form.get('job_title', '').strip()
        job_description = request.form.get('job_description', '').strip()
        company_profile = request.form.get('company_profile', '').strip()
        requirements = request.form.get('requirements', '').strip()
        benefits = request.form.get('benefits', '').strip()
        
        # Validate: require at least title and description
        if not job_title or not job_description:
            return jsonify({'error': 'Please provide at least a Job Title and Description.'}), 400
        
        # Run prediction
        if model_loaded:
            # REAL AI PREDICTION (Bi-LSTM model)
            probability, cleaned_text = predict_fraud(
                job_title, job_description, company_profile, requirements, benefits
            )
            prediction_mode = 'AI Model (Bi-LSTM)'
        else:
            # DEMO MODE (rule-based fallback)
            probability, cleaned_text = demo_predict(
                job_title, job_description, company_profile, requirements, benefits
            )
            prediction_mode = 'Demo Mode (Train model for AI predictions)'
        
        # Determine verdict and risk level
        if probability >= 0.75:
            verdict = 'FRAUDULENT'
            risk_level = 'HIGH RISK'
            risk_class = 'danger'
            risk_emoji = '🚨'
        elif probability >= 0.50:
            verdict = 'LIKELY FRAUDULENT'
            risk_level = 'MEDIUM RISK'
            risk_class = 'warning'
            risk_emoji = '⚠️'
        elif probability >= 0.30:
            verdict = 'SUSPICIOUS'
            risk_level = 'LOW RISK'
            risk_class = 'info'
            risk_emoji = '🔍'
        else:
            verdict = 'LIKELY LEGITIMATE'
            risk_level = 'APPEARS SAFE'
            risk_class = 'success'
            risk_emoji = '✅'
        
        # Build response
        return jsonify({
            'verdict': verdict,
            'probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'risk_class': risk_class,
            'risk_emoji': risk_emoji,
            'prediction_mode': prediction_mode,
            'model_loaded': model_loaded,
            'word_count': len(cleaned_text.split()) if cleaned_text != 'demo_mode' else 'N/A'
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500


@app.route('/model-info')
def model_info():
    """Returns information about the loaded model"""
    eval_results = {}
    eval_path = 'models/evaluation_results.json'
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
    
    return jsonify({
        'model_loaded': model_loaded,
        'model_path': MODEL_PATH,
        'config': config,
        'evaluation': eval_results
    })


@app.route('/about')
def about():
    return render_template('about.html')


# ============================================================
# RUN THE SERVER
# ============================================================
if __name__ == '__main__':
    print(f"\n🌐 Server starting at: http://127.0.0.1:5000")
    print(f"   Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
