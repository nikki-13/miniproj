# Troubleshooting Guide

This guide addresses common issues you might encounter when setting up and running the X-Ray Insight Lab server.

## Installation Issues

### ModuleNotFoundError: No module named 'uvicorn' (or any other package)

**Problem**: Python cannot find the required packages.

**Solutions**:

1. Use the improved run.py script which will automatically check for dependencies and attempt to install them:
   ```
   ./run.py
   ```
   
   The script will provide detailed diagnostic information and attempt to fix common issues.

2. Make sure you've installed all dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Try installing the specific missing package:
   ```
   pip install uvicorn
   ```

4. Use the setup script which handles dependency installation:
   ```
   ./setup.sh
   ```

5. Check if you're using the correct Python environment (if you're using virtual environments).

6. If you have multiple Python installations, try specifying the Python version:
   ```
   python3 -m pip install uvicorn
   ```

### Permission Denied when running setup.sh

**Problem**: The setup script doesn't have execution permissions.

**Solution**:
```
chmod +x setup.sh
```

## Server Startup Issues

### Address already in use

**Problem**: Port 8000 is already being used by another application.

**Solution**:
1. Find and stop the process using port 8000, or
2. Modify the port in `run.py`:
   ```python
   uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
   ```

### FileNotFoundError: Model file not found

**Problem**: The server cannot find the PyTorch model file.

**Solution**:
1. Ensure the model file exists at `../src/model/swin_pneumonia_model.pth` relative to the server directory.
2. If the model is in a different location, update the `MODEL_PATH` in `app/model.py`.

## Prediction Issues

### Error during prediction

**Problem**: The server encounters an error when processing an uploaded image.

**Solutions**:
1. Ensure you're uploading a valid image file (JPG, PNG, JPEG).
2. Check the server logs for specific error messages.
3. Verify that the model file is correctly loaded.

### CORS errors in browser console

**Problem**: The frontend cannot communicate with the backend due to CORS restrictions.

**Solution**:
The server is already configured to allow CORS. Make sure:
1. The frontend is making requests to the correct URL (http://localhost:8000).
2. The CORS middleware in `app/main.py` is properly configured.

## Performance Issues

### Slow prediction times

**Problem**: The model takes a long time to process images.

**Solutions**:
1. If available, ensure you're using a machine with a GPU.
2. For CPU-only machines, consider using a smaller model or optimizing the existing one.
3. Check if other processes are consuming system resources.

## Other Issues

If you encounter issues not covered in this guide:

1. Check the server logs for error messages.
2. Look for error messages in the browser console (for frontend issues).
3. Try restarting both the frontend and backend servers.
4. Ensure all dependencies are correctly installed.
