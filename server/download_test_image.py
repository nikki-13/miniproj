#!/usr/bin/env python3

import os
import requests
import sys
from pathlib import Path

# URLs for sample X-ray images
SAMPLE_IMAGES = {
    "normal": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000001_000.png",
    "pneumonia": "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000099_002.png"
}

def download_image(url, filename):
    """Download an image from a URL"""
    try:
        print(f"Downloading {filename} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create the test_images directory if it doesn't exist
        os.makedirs("test_images", exist_ok=True)
        
        # Save the image
        file_path = os.path.join("test_images", filename)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Image saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None

def main():
    """Download sample X-ray images for testing"""
    print("Downloading sample X-ray images for testing")
    print("------------------------------------------")
    
    downloaded_files = []
    
    for image_type, url in SAMPLE_IMAGES.items():
        filename = f"{image_type}_xray.png"
        file_path = download_image(url, filename)
        if file_path:
            downloaded_files.append(file_path)
    
    if downloaded_files:
        print("\nDownloaded images:")
        for file_path in downloaded_files:
            print(f"  - {file_path}")
        
        print("\nTo test the server with these images, run:")
        print(f"python test_server.py {downloaded_files[0]}")
    else:
        print("\nNo images were downloaded. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
