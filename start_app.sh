#!/bin/bash

# X-Ray Insight Lab Startup Script

echo "Starting X-Ray Insight Lab..."
echo "----------------------------"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if required commands exist
if ! command_exists npm; then
    echo "❌ npm is not installed. Please install Node.js and npm."
    exit 1
fi

if ! command_exists python3 && ! command_exists python; then
    echo "❌ Python is not installed. Please install Python 3."
    exit 1
fi

# Determine Python command
PYTHON_CMD="python3"
if ! command_exists python3; then
    PYTHON_CMD="python"
fi

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install frontend dependencies."
        exit 1
    fi
    echo "✅ Frontend dependencies installed"
fi

# Install backend dependencies if needed
echo "Setting up backend..."
cd server
if [ ! -f "venv/bin/activate" ] && [ ! -d "venv/Scripts" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo "⚠️ Could not create virtual environment. Continuing with system Python..."
    else
        echo "✅ Virtual environment created"
    fi
fi

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -f "venv/Scripts/activate" ]; then
    echo "Activating virtual environment..."
    source venv/Scripts/activate
    echo "✅ Virtual environment activated"
fi

# Install backend dependencies
echo "Installing backend dependencies..."
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies could not be installed. Trying individual installations..."
    $PYTHON_CMD -m pip install fastapi uvicorn python-multipart torch torchvision Pillow requests timm
fi
echo "✅ Backend dependencies installed"

# Download test images if they don't exist
if [ ! -d "test_images" ]; then
    echo "Downloading test images..."
    $PYTHON_CMD download_test_image.py
    echo "✅ Test images downloaded"
fi

# Start the backend server in the background
echo "Starting backend server..."
$PYTHON_CMD run.py &
BACKEND_PID=$!
echo "✅ Backend server started (PID: $BACKEND_PID)"

# Go back to the main directory
cd ..

# Start the frontend
echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "X-Ray Insight Lab is now running!"
echo "--------------------------------"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173 (or check the URL in the terminal output)"
echo ""
echo "API Documentation:"
echo "- Swagger UI: http://localhost:8000/docs"
echo "- ReDoc: http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped"
    exit 0
}

# Set up trap to catch Ctrl+C
trap cleanup INT

# Wait for user to press Ctrl+C
wait
