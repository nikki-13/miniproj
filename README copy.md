# X-Ray Insight Lab

X-Ray Insight Lab is a web application that uses AI to analyze chest X-ray images for signs of pneumonia. The application consists of a React frontend and a FastAPI backend that uses a pre-trained PyTorch model for prediction.

## Project Structure

- `src/` - Frontend React application
- `server/` - FastAPI backend server
- `src/model/` - Contains the pre-trained PyTorch model

## Setup and Installation

### Backend Setup

#### Option 1: Using the Setup Script (Recommended)

1. Navigate to the server directory:
   ```
   cd server
   ```

2. Run the setup script:
   ```
   ./setup.sh
   ```
   
   This script will:
   - Check if Python is installed
   - Create and activate a virtual environment (optional)
   - Install all required dependencies
   - Verify the model file location

#### Option 2: Manual Setup

1. Navigate to the server directory:
   ```
   cd server
   ```

2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   If you encounter any issues, you can try installing the dependencies individually:
   ```
   pip install fastapi uvicorn python-multipart torch torchvision Pillow requests
   ```

3. Ensure the model file is in the correct location:
   - The model should be at `src/model/swin_pneumonia_model.pth`

### Frontend Setup

1. Install the required Node.js dependencies:
   ```
   npm install
   ```

## Running the Application

### Option 1: Using the All-in-One Startup Script (Recommended)

We've provided a convenient script that sets up and starts both the frontend and backend in one command:

```
./start_app.sh
```

This script will:
- Install all required dependencies for both frontend and backend
- Create a virtual environment for Python (optional)
- Download sample X-ray images for testing
- Start the backend server
- Start the frontend development server
- Provide URLs to access the application
- Handle graceful shutdown of both servers when you press Ctrl+C

### Option 2: Manual Startup

If you prefer to start the components separately:

#### Start the Backend Server

1. Navigate to the server directory:
   ```
   cd server
   ```

2. Run the FastAPI server:
   ```
   ./run.py
   ```

3. Download sample X-ray images for testing:
   ```
   ./download_test_image.py
   ```

4. Test the server with the downloaded images:
   ```
   python test_server.py test_images/normal_xray.png
   ```

#### Start the Frontend Application

1. In a new terminal, run the frontend development server:
   ```
   npm run dev
   ```

2. Open your browser and navigate to `http://localhost:5173` to use the application.

## Using the Application

1. Navigate to the Upload page
2. Upload a chest X-ray image by dragging and dropping or clicking to select a file
3. The AI model will analyze the image and display the prediction results
4. The results include:
   - Prediction (Normal or Pneumonia)
   - Confidence percentage
   - Probabilities for each class

## API Documentation

Once the backend server is running, you can access the auto-generated API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

If you encounter any issues while setting up or running the application, please refer to the [Troubleshooting Guide](server/TROUBLESHOOTING.md).

## API Endpoints

### GET /
- Returns a simple message to confirm the API is running

### POST /predict
- Accepts an X-ray image file upload
- Returns prediction results including:
  - Filename
  - Prediction (Normal or Pneumonia)
  - Confidence percentage
  - Probabilities for each class

## Technologies Used

### Frontend
- React
- TypeScript
- Vite
- TailwindCSS
- Shadcn UI Components

### Backend
- FastAPI
- PyTorch
- Python

## Model Information

The application uses a pre-trained Swin Transformer model for pneumonia detection from chest X-ray images. The model has been trained on a dataset of chest X-ray images and can classify images as either "Normal" or "Pneumonia".
