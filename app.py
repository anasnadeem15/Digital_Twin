from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained models
with open('trained_model.pkl', 'rb') as file:
    lin_reg = pickle.load(file)

with open('power_classification_model.pkl', 'rb') as file:
    log_reg = pickle.load(file)

# Function to parse G-code
def parse_gcode(gcode):
    feed_rate = 0  # Default feed rate
    spindle_speed = 0  # Default spindle speed
    depth_of_cut = 0  # Default depth of cut

    for line in gcode.split('\n'):
        try:
            if 'F' in line:  # Extract Feed Rate
                feed_rate_str = line.split('F')[1].split()[0]
                feed_rate = float(feed_rate_str) if feed_rate_str.replace('.', '', 1).isdigit() else 0
            if 'S' in line:  # Extract Spindle Speed
                spindle_speed_str = line.split('S')[1].split()[0]
                spindle_speed = float(spindle_speed_str) if spindle_speed_str.replace('.', '', 1).isdigit() else 0
            if 'Z' in line and 'G1' in line:  # Extract Depth of Cut (Z-axis movement)
                z_values = [float(val.replace('Z', '')) for val in line.split() if 'Z' in val and val.replace('Z', '').replace('.', '', 1).isdigit()]
                if z_values:
                    depth_of_cut = abs(z_values[0] - z_values[1]) if len(z_values) > 1 else 0
        except (IndexError, ValueError):
            # Skip lines that cannot be parsed
            continue

    return feed_rate, spindle_speed, depth_of_cut

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle manual input form
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    feed_rate = float(request.form.get('feed_rate', 0))  # Default to 0 if missing
    depth_of_cut = float(request.form.get('depth_of_cut', 0))  # Default to 0 if missing
    spindle_speed = float(request.form.get('spindle_speed', 0))  # Default to 0 if missing

    # Predict power consumption
    input_data = [[feed_rate, depth_of_cut, spindle_speed]]
    power_consumption = lin_reg.predict(input_data)[0]

    # Classify power source
    power_source = log_reg.predict(input_data)[0]
    power_source = "Solar" if power_source == 1 else "K-Electric"

    return render_template('index.html', 
                           power_prediction=power_consumption,
                           power_source=power_source)

# Route to handle G-code file upload
@app.route('/process_gcode', methods=['POST'])
def process_gcode():
    gcode = request.json['gcode']
    
    # Parse G-code to extract parameters
    feed_rate, spindle_speed, depth_of_cut = parse_gcode(gcode)
    
    # Predict power consumption
    input_data = [[feed_rate, depth_of_cut, spindle_speed]]
    power_consumption = lin_reg.predict(input_data)[0]
    
    # Classify power source
    power_source = log_reg.predict(input_data)[0]
    power_source = "Solar" if power_source == 1 else "K-Electric"
    
    return jsonify({
        "power_consumption": f"{power_consumption:.2f} watts",
        "power_source": power_source
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)