import requests
import sys
import os
from pathlib import Path

def test_server_connection():
    """Test if the server is running and responding"""
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("✅ Server is running and responding")
            return True
        else:
            print(f"❌ Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on http://localhost:8000")
        return False

def test_prediction_endpoint(image_path=None):
    """Test the prediction endpoint with a sample image"""
    if not image_path:
        print("No image path provided. Skipping prediction test.")
        return False
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found at {image_path}")
        return False
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction endpoint is working")
            print("\nPrediction Result:")
            print(f"  - Filename: {result.get('filename')}")
            print(f"  - Prediction: {result.get('prediction')}")
            print(f"  - Confidence: {result.get('confidence')}")
            print("\nProbabilities:")
            for class_name, prob in result.get('probabilities', {}).items():
                print(f"  - {class_name}: {prob}")
            return True
        else:
            print(f"❌ Prediction endpoint returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during prediction test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing X-Ray Insight FastAPI Server\n")
    
    # Test server connection
    if not test_server_connection():
        print("\nPlease make sure the server is running with 'python run.py'")
        sys.exit(1)
    
    # Test prediction endpoint with a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_prediction_endpoint(image_path)
    else:
        print("\nTo test the prediction endpoint, provide an image path:")
        print("python test_server.py path/to/xray_image.jpg")
