from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import pickle
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'nc', 'txt', 'gcode', 'tap'}
app.config.update({
    'UPLOAD_FOLDER': UPLOAD_FOLDER,
    'MAX_CONTENT_LENGTH': 10 * 1024 * 1024,  # 10MB
    'SECRET_KEY': os.urandom(24)
})
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ML models with version check
try:
    MODELS = {
        'power_model': pickle.load(open('models/trained_model.pkl', 'rb')),
        'scaler': pickle.load(open('models/scaler.pkl', 'rb')),
        'source_model': pickle.load(open('models/power_classification_model.pkl', 'rb'))
    }
    logger.info("ML models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    MODELS = None

TARIFFS = {
    'solar': 12.5,  # per kWh
    'grid': 25.0    # per kWh
}

def parse_gcode(gcode):
    feed = speed = depth = 0.0
    x_values = []
    z_values = []
    
    for line in gcode.split('\n'):
        line = line.strip().upper()
        
        # Skip comments and program numbers
        if any(line.startswith(c) for c in ('(', '%', 'O', 'N', '#')):
            continue
            
        # Extract feed rate (handles negative values)
        if 'F' in line:
            try:
                f_val = float(line.split('F')[1].split()[0])
                feed = max(feed, abs(f_val))
            except (ValueError, IndexError):
                pass
                
        # Extract spindle speed (ignore CSS modes)
        if 'S' in line and 'G96' not in line:
            try:
                s_val = float(line.split('S')[1].split()[0])
                speed = max(speed, abs(s_val))
            except (ValueError, IndexError):
                pass
                
        # Extract coordinates (handles negative values)
        if any(cmd in line for cmd in ['G00', 'G01', 'G02', 'G03']):
            for param in line.split():
                if param.startswith('X'):
                    try:
                        x_val = float(param[1:])
                        x_values.append(x_val)
                    except ValueError:
                        continue
                elif param.startswith('Z'):
                    try:
                        z_val = float(param[1:])
                        z_values.append(z_val)
                    except ValueError:
                        continue
    
    # Calculate depth of cut (absolute differences)
    if x_values:
        depth = (max(x_values) - min(x_values)) / 2  # For turning ops
    elif z_values:
        depth = max(z_values) - min(z_values)
    
    # If spindle speed is still zero, set a default (e.g., 1000 RPM)
    if speed == 0.0:
        logger.info("Spindle speed not found in G-code; using default value of 1000 RPM.")
        speed = 1000.0
    
    # Optionally, if feed is zero, you might set a default value (if that makes sense for your process)
    if feed == 0.0:
        logger.info("Feed rate not found in G-code; using default value of 0.1.")
        feed = 0.1

    feed = round(feed, 4)
    speed = round(speed, 0)
    depth = round(abs(depth), 4)
    logger.info(f"Parsed G-code parameters -> Feed: {feed}, Speed: {speed}, Depth: {depth}")
    return feed, speed, depth

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("10/minute")
def handle_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        with open(save_path, 'r') as f:
            gcode = f.read()
        
        feed, speed, depth = parse_gcode(gcode)
        logger.info(f"Processed file: {filename} | Feed: {feed} | Speed: {speed} | Depth: {depth}")
        
        return jsonify({
            'success': True,
            'parameters': {
                'feed_rate': feed,
                'spindle_speed': speed,
                'depth_of_cut': depth
            }
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
@limiter.limit("20/minute")
def handle_prediction():
    try:
        data = request.get_json()
        # Ensure positive values even if sent negative
        feed = abs(float(data.get('feed_rate', 0)))
        speed = abs(float(data.get('spindle_speed', 0)))
        depth = abs(float(data.get('depth_of_cut', 0)))
        
        logger.info(f"Received for prediction -> Feed: {feed}, Speed: {speed}, Depth: {depth}")
        
        if any(v <= 0 for v in [feed, speed, depth]):
            raise ValueError("Parameters must be positive numbers")
        
        # Feature scaling using your scaler model
        X_scaled = MODELS['scaler'].transform([[feed, depth, speed]])
        
        # Predictions
        power = round(float(MODELS['power_model'].predict(X_scaled)[0]), 2)
        source = "Solar" if MODELS['source_model'].predict(X_scaled)[0] == 1 else "Grid"
        
        # Tool life calculation (Taylor's tool life equation approximation)
        tool_life = (200 / speed) * (1 / (feed ** 0.8)) * (1 / (depth ** 0.4)) * 60
        tool_life = round(tool_life, 1)
        
        # Cost calculations
        operation_time = tool_life / 60  # Hours
        energy_consumed = (power * operation_time) / 1000  # kWh
        
        costs = {
            'solar': round(energy_consumed * TARIFFS['solar'], 2),
            'grid': round(energy_consumed * TARIFFS['grid'], 2)
        }
        
        return jsonify({
            'power': power,
            'source': source,
            'tool_life': tool_life,
            'costs': costs,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
