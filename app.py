"""
Flask API Server for Strawberry Leaf Health Analysis
Optimized for Render deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from strawberry_leaf_analyzer import StrawberryLeafAnalyzer
import base64
import traceback
import numpy as np
import os

app = Flask(__name__)

# Configure CORS - allow requests from your GitHub Pages site
CORS(app, origins=[
    "https://zaaditto.github.io",
    "http://localhost:3000",  # For local testing
    "http://127.0.0.1:3000"
])

# Initialize analyzer
analyzer = StrawberryLeafAnalyzer()


def convert_to_native(obj):
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    This fixes the 'Object of type bool_ is not JSON serializable' error.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native(i) for i in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    else:
        return obj


@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Strawberry Leaf Health Analyzer API',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'analyze': '/api/analyze (POST)',
            'analyze_file': '/api/analyze-file (POST)'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Strawberry Leaf Analyzer API is running'
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_leaf():
    """
    Analyze a strawberry leaf image
    
    Expects JSON with:
    - image: base64 encoded image string (with or without data URL prefix)
    
    Returns:
    - Analysis results with all health indicators
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please send base64 encoded image.'
            }), 400
        
        # Get base64 image
        image_data = data['image']
        
        # Analyze the image
        results = analyzer.analyze(base64_string=image_data)
        
        # Convert all NumPy types to Python native types
        results = convert_to_native(results)
        
        return jsonify(results)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/analyze-file', methods=['POST'])
def analyze_leaf_file():
    """
    Analyze a strawberry leaf image from file upload
    
    Expects:
    - file: Image file in multipart form data
    
    Returns:
    - Analysis results with all health indicators
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read and encode the file
        image_bytes = file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Determine mime type
        mime_type = file.content_type or 'image/jpeg'
        base64_with_prefix = f"data:{mime_type};base64,{base64_image}"
        
        # Analyze the image
        results = analyzer.analyze(base64_string=base64_with_prefix)
        
        # Convert all NumPy types to Python native types
        results = convert_to_native(results)
        
        return jsonify(results)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }), 500


if __name__ == '__main__':
    # This block is only for local development
    # In production, Gunicorn will run the app
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("🍓 Strawberry Leaf Health Analyzer API")
    print("=" * 60)
    print(f"Server starting on http://localhost:{port}")
    print("")
    print("Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /api/health    - Health check")
    print("  POST /api/analyze   - Analyze base64 image (JSON)")
    print("  POST /api/analyze-file - Analyze uploaded file (multipart)")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port)
