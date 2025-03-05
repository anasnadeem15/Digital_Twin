from flask import Flask, render_template, request
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained models and scaler
with open(os.path.join('trained_model.pkl'), 'rb') as file:
    lin_reg = pickle.load(file)

with open(os.path.join('power_classification_model.pkl'), 'rb') as file:
    log_reg = pickle.load(file)

with open(os.path.join('scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    feed_rate = float(request.form['feed_rate'])
    depth_of_cut = float(request.form['depth_of_cut'])
    spindle_speed = float(request.form['spindle_speed'])

    # Scale the input data
    input_data = scaler.transform([[feed_rate, depth_of_cut, spindle_speed]])

    # Predict power consumption
    power_prediction = lin_reg.predict(input_data)[0]

    # Predict power source
    power_source_prediction = log_reg.predict(input_data)[0]
    power_source = "Solar" if power_source_prediction == 1 else "K-Electric"

    # Return the results
    return render_template('index.html', 
                           power_prediction=power_prediction,
                           power_source=power_source)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)