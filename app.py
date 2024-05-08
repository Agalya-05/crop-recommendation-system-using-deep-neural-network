from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('crop_recommendation_model.h5')
scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Define crop dictionary
crop_dict = {
    0: 'Rice', 1: 'Maize', 2: 'Jute', 3: 'Cotton', 4: 'Coconut', 5: 'Papaya', 6: 'Orange',
    7: 'Apple', 8: 'Muskmelon', 9: 'Watermelon', 10: 'Grapes', 11: 'Mango', 12: 'Banana',
    13: 'Pomegranate', 14: 'Lentil', 15: 'Blackgram', 16: 'Mungbean', 17: 'Mothbeans',
    18: 'Pigeonpeas', 19: 'Kidneybeans', 20: 'Chickpea', 21: 'Coffee'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Scale the input features
    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)
    predicted_class_index = np.argmax(prediction)
    crop_name = crop_dict[predicted_class_index]

    return render_template('index.html', prediction_text=f'The recommended crop is: {crop_name}')

if __name__ == '__main__':
    app.run(debug=True)
