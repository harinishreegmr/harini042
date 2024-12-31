from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import load_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations


app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16MB
app.secret_key = 'your-secret-key'  # Set a secret key for session management

# Simulated users database (for demo purposes)
users = {
    'user1': {'password': 'password123'},
    'user2': {'password': 'mypassword'},
}

# Load the pre-trained model
model = load_model('crop_disease_model.h5')

# Load the disease details CSV
disease_details_df = pd.read_csv('disease_details.csv')

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to predict the disease based on the uploaded image
def predict_disease(uploaded_image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(uploaded_image_path, target_size=(224, 224))  # Resize to match model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize the image (scale to [0, 1])
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Predict the disease
        prediction = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        
        # Map the predicted class index to the disease name
        disease_name = disease_details_df.iloc[predicted_class_idx]['disease_name']
        
        return disease_name
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown"

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and users[username]['password'] == password:
            session['username'] = username
            return redirect(url_for('index'))
        return "Invalid credentials. Please try again."
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username not in users:
            users[username] = {'password': password}
            session['username'] = username
            return redirect(url_for('index'))
        return "Username already exists. Please try a different one."
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/scan', methods=['GET', 'POST'])
def scan():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in request."
        
        file = request.files['file']
        
        if file.filename == '':
            return "No file selected."
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the disease
            disease_name = predict_disease(file_path)

            # Redirect to result page with predicted disease name
            return redirect(url_for('result', disease=disease_name))
    
    return render_template('scan.html')

@app.route('/result')
def result():
    # Ensure no extra spaces in column names
    disease_details_df.columns = disease_details_df.columns.str.strip()

    disease = request.args.get('disease')  # Get the disease from the URL query parameter
    
    # Check if the disease exists in the DataFrame
    if disease not in disease_details_df['disease_name'].values:
        # Handle case when the disease is not found
        print("Disease not found")
        return "Disease not found in the database."
    else:
        # Retrieve the disease details
        disease_info = disease_details_df[disease_details_df['disease_name'] == disease].iloc[0]
        
        return render_template(
            'result.html', 
            disease=disease,
            cause=disease_info['cause'],
            effect=disease_info['effect'],
            prevention=disease_info['prevention'],
            pesticide=disease_info['pesticides']
        )

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5002)
