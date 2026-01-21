#!/usr/bin/env python3
"""
Flask Web Demo for Pose Estimation Pipeline
Provides a web interface to test the pose estimation pipeline.
"""

import os
import json
import tempfile
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'pose_estimation_uploads'
OUTPUT_FOLDER = Path(tempfile.gettempdir()) / 'pose_estimation_outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'jpg', 'jpeg', 'png'}

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_FOLDER'] = str(OUTPUT_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Generate unique filename
    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{unique_id}{ext}"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': unique_filename,
        'file_id': unique_id
    })


@app.route('/api/process', methods=['POST'])
def process_file():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Import processing modules
        from pose_estimation import PoseEstimator, process_video, process_image
        import yaml
        
        # Load default config
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        ext = os.path.splitext(filename)[1].lower()
        output_dir = app.config['OUTPUT_FOLDER']
        
        if ext in {'.jpg', '.jpeg', '.png'}:
            result = process_image(filepath, output_dir, config)
        else:
            result = process_video(filepath, output_dir, config, save_video=True)
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/status')
def status():
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'upload_folder': str(UPLOAD_FOLDER),
        'output_folder': str(OUTPUT_FOLDER)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
