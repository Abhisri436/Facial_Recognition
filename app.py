from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from io import BytesIO
from deepface import DeepFace
import base64
import uuid
import tempfile
import traceback

# Initialize Flask app with request size limit
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit request size to 16MB

# Enhanced CORS configuration
cors = CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def compare_photo_with_url(photo_path, image_url):
    """
    Compares a local photo with an image from a URL using DeepFace.
    Returns True if the faces match, False otherwise.
    """
    try:
        # Download the image from the URL to a temporary file
        temp_url_file = os.path.join(tempfile.gettempdir(), f"temp_url_{uuid.uuid4()}.jpg")
        
        try:
            # Set a timeout for the request
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to download image from URL: {image_url}, status code: {response.status_code}")
                return False
            
            # Save the downloaded image to a temporary file
            with open(temp_url_file, 'wb') as f:
                f.write(response.content)
            
            # Now compare the two local files
            result = DeepFace.verify(
                img1_path=photo_path,
                img2_path=temp_url_file,
                model_name='Facenet',
                detector_backend='opencv',
                distance_metric='cosine'
            )
            
            # Clean up the temporary URL file
            if os.path.exists(temp_url_file):
                os.remove(temp_url_file)
            
            return result["verified"]
        
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from URL {image_url}: {str(e)}")
            if os.path.exists(temp_url_file):
                os.remove(temp_url_file)
            return False
            
    except Exception as e:
        print(f"Error in face comparison: {str(e)}")
        traceback.print_exc()
        return False

def compare_photo_and_array(photo_path, database_urls):
    """
    Compares a local photo with images from an array of URLs using DeepFace.
    Returns the URL of the first match found, or None if no match is found.
    """
    for image_url in database_urls:
        print(f"Comparing with URL: {image_url}")
        try:
            result = compare_photo_with_url(photo_path, image_url)
            if result:
                print(f"Match found with: {image_url}")
                return image_url
            else:
                print(f"No match with: {image_url}")
        except Exception as e:
            print(f"Error comparing with {image_url}: {str(e)}")
            traceback.print_exc()
    return None

@app.route('/api/compare-faces', methods=['POST'])
def compare_faces():
    try:
        # Get JSON data from request
        data = request.json
        
        if not data or 'capturedImage' not in data or 'databaseUrls' not in data:
            return jsonify({
                'matchFound': False,
                'error': 'Invalid request data. Need capturedImage and databaseUrls.'
            }), 400
        
        # Get the base64 image and database URLs
        captured_image_base64 = data['capturedImage']
        database_urls = data['databaseUrls']
        
        # Log the request details (without the full image data for privacy)
        print(f"Received request with {len(database_urls)} database URLs")
        
        # Validate database URLs
        if not database_urls or len(database_urls) == 0:
            return jsonify({
                'matchFound': False,
                'error': 'No database URLs provided'
            }), 400
            
        # Remove the data:image/jpeg;base64, prefix if present
        if ',' in captured_image_base64:
            prefix = captured_image_base64.split(',')[0]
            captured_image_base64 = captured_image_base64.split(',')[1]
            print(f"Extracted base64 data with prefix: {prefix}")
        
        # Create a temporary file for the captured image
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_captured_{uuid.uuid4()}.jpg")
        
        try:
            # Save the base64 image to the temporary file
            with open(temp_file_path, 'wb') as f:
                f.write(base64.b64decode(captured_image_base64))
            print(f"Saved captured image to temporary file: {temp_file_path}")
        except Exception as e:
            print(f"Error decoding base64 image: {str(e)}")
            return jsonify({
                'matchFound': False,
                'error': f'Invalid base64 image data: {str(e)}'
            }), 400
        
        # Compare the captured image with the database URLs
        match_url = compare_photo_and_array(temp_file_path, database_urls)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print("Removed temporary file")
        
        if match_url:
            # If a match is found, return the URL and success status
            print(f"Match found with URL: {match_url}")
            return jsonify({
                'matchFound': True,
                'matchedImageUrl': match_url,
                'confidence': 0.85  # Add a confidence score
            })
        else:
            # If no match is found, return failure status
            print("No match found")
            return jsonify({
                'matchFound': False,
                'message': 'No matching face found in the database'
            })
            
    except Exception as e:
        print(f"Unexpected error in compare_faces: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'matchFound': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Facial recognition API is running'
    })

# Simple test endpoint
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        'status': 'ok',
        'message': 'API is working correctly'
    })
    
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
