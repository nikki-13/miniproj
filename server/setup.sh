#!/bin/bash

# X-Ray Insight Lab Server Setup Script

echo "Setting up X-Ray Insight Lab Server..."
echo "-------------------------------------"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

echo "✅ Python 3 is installed"

# Create a virtual environment (optional)
echo "Creating a virtual environment (optional)..."
python3 -m venv venv 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created"
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️ Could not create virtual environment. Continuing with system Python..."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "⚠️ Some dependencies could not be installed. Trying individual installations..."
    
    # Try installing dependencies individually
    pip install fastapi
    pip install uvicorn
    pip install python-multipart
    pip install torch
    pip install torchvision
    pip install Pillow
    pip install requests
    
    echo "✅ Individual dependency installation completed"
fi

# Check if model file exists
MODEL_PATH="../src/model/swin_pneumonia_model.pth"
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Model file found at $MODEL_PATH"
else
    echo "⚠️ Model file not found at $MODEL_PATH. Please ensure the model file is in the correct location."
fi

echo "-------------------------------------"
echo "Setup completed!"
echo ""
echo "To run the server, use:"
echo "python run.py"
echo ""
echo "To test the server with a sample image, use:"
echo "python test_server.py path/to/xray_image.jpg"
