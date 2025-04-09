# X-Ray Insight FastAPI Server

This is the server-side component of the X-Ray Insight application, built with FastAPI. It provides an API endpoint for X-ray image prediction using a pre-trained PyTorch model.

## Setup

1. Make sure you have Python 3.8+ installed

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the model file is in the correct location:
   - The model should be at `../src/model/swin_pneumonia_model.pth` relative to the server directory

## Running the Server

Run the server using one of the following methods:

### Method 1: Using the executable script (Recommended)
```
./run.py
```

### Method 2: Using Python
```
python run.py
```

The improved run.py script will:
- Check if you're in the correct directory
- Verify that uvicorn is installed (and attempt to install it if missing)
- Display helpful diagnostic information
- Start the FastAPI server at `http://localhost:8000`

If you encounter any issues, the script will provide detailed error messages to help troubleshoot the problem.

## Testing the Server

### Download Test Images

We've provided a script to download sample X-ray images for testing:

```
./download_test_image.py
```

This script will:
- Download sample normal and pneumonia X-ray images
- Save them to a `test_images` directory
- Provide instructions on how to test the server with these images

### Test the Server

Once you have test images, you can test the server using:

```
python test_server.py test_images/normal_xray.png
```

or

```
python test_server.py test_images/pneumonia_xray.png
```

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

## API Documentation

Once the server is running, you can access the auto-generated API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Troubleshooting

If you encounter any issues while setting up or running the server, please refer to the [Troubleshooting Guide](TROUBLESHOOTING.md).
