from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from io import BytesIO
from deepface import DeepFace
import base64
import uuid
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def compare_photo_with_url(photo_path, image_url):
    """
    Compares a local photo with an image from a URL using DeepFace.
    Returns True if the faces match, False otherwise.
    """
    try:
        result = DeepFace.verify(
            img1_path=photo_path,
            img2_path=image_url,
            model_name='VGG-Face',  # You can use other models like 'Facenet', 'OpenFace', 'DeepFace', etc.
            detector_backend='opencv',
            distance_metric='cosine'
        )
        return result["verified"]
    except Exception as e:
        print(f"Error in face comparison: {str(e)}")
        return False

def compare_photo_and_array(photo_path, database_urls):
    """
    Compares a local photo with images from an array of URLs using DeepFace.
    Returns the URL of the first match found, or None if no match is found.
    """
    for image_url in database_urls:
        print(f"Comparing with: {image_url}")
        try:
            result = compare_photo_with_url(photo_path, image_url)
            if result:
                print(f"Match found with: {image_url}")
                return image_url
        except Exception as e:
            print(f"Error comparing with {image_url}: {str(e)}")
    return None

@app.route('/api/compare-faces', methods=['POST'])
def compare_faces():
    try:
        # Get JSON data from request
        data = request.json
        
        if not data or 'capturedImage' not in data or 'databaseUrls' not in data:
            return jsonify({'error': 'Invalid request data. Need capturedImage and databaseUrls.'}), 400
        
        # Get the base64 image and database URLs
        captured_image_base64 = data['capturedImage']
        database_urls = data['databaseUrls']
        
        # Remove the data:image/jpeg;base64, prefix if present
        if ',' in captured_image_base64:
            captured_image_base64 = captured_image_base64.split(',')[1]
        
        # Create a temporary file for the captured image
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"temp_captured_{uuid.uuid4()}.jpg")
        
        # Save the base64 image to the temporary file
        with open(temp_file_path, 'wb') as f:
            f.write(base64.b64decode(captured_image_base64))
        
        # Compare the captured image with the database URLs
        match_url = compare_photo_and_array(temp_file_path, database_urls)
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        if match_url:
            # If a match is found, return the URL and success status
            return jsonify({
                'matchFound': True,
                'matchedImageUrl': match_url
            })
        else:
            # If no match is found, return failure status
            return jsonify({
                'matchFound': False
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
