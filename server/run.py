#!/usr/bin/env python3

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import uvicorn
        print("✅ uvicorn is installed")
        return True
    except ImportError:
        print("❌ uvicorn is not installed")
        
        # Try to install uvicorn
        print("Attempting to install uvicorn...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
            print("✅ uvicorn installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install uvicorn: {e}")
            
            # Provide detailed instructions
            print("\nPlease install uvicorn manually:")
            print("1. Run: pip install uvicorn")
            print("2. If that doesn't work, try: python -m pip install uvicorn")
            print("3. If you're using a virtual environment, make sure it's activated")
            
            return False

def run_server():
    """Run the FastAPI server"""
    try:
        import uvicorn
        print("Starting X-Ray Insight FastAPI server...")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        
        # Check if the app module can be imported
        try:
            import app.main
            print("✅ app.main module can be imported")
        except ImportError:
            print("❌ Cannot import app.main module")
            print("Make sure you're running this script from the server directory")

if __name__ == "__main__":
    print("X-Ray Insight FastAPI Server")
    print("----------------------------")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Print current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the server directory
    if not os.path.exists("app") or not os.path.isdir("app"):
        print("❌ Error: 'app' directory not found")
        print("Make sure you're running this script from the server directory")
        sys.exit(1)
    
    # Check dependencies
    if check_dependencies():
        # Run the server
        run_server()
    else:
        sys.exit(1)
