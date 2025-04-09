import io
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
from pathlib import Path
import timm

# Get the absolute path to the model file
# Try different possible locations
possible_paths = [
    Path("../src/model/swin_pneumonia_model.pth").resolve(),
    Path("../../src/model/swin_pneumonia_model.pth").resolve(),
    Path(os.path.join(os.path.dirname(__file__), "../../src/model/swin_pneumonia_model.pth")).resolve()
]

# Find the first path that exists
MODEL_PATH = None
for path in possible_paths:
    if path.exists():
        MODEL_PATH = path
        print(f"Found model at: {MODEL_PATH}")
        break

# If model not found, raise error
if MODEL_PATH is None:
    print("Error: Model file not found. Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    model = None
else:
    # Load the model (this will be executed when the module is imported)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create the model architecture using timm
        try:
            # Create the same model architecture as used in training
            model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)
            # Load the state dict
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()  # Set the model to evaluation mode
            print("Model loaded successfully using timm")
        except Exception as e:
            print(f"Failed to load with timm: {e}")
            try:
                # Fallback method: Direct load
                model = torch.load(MODEL_PATH, map_location=device)
                model.eval()
                print("Model loaded using direct torch.load()")
            except Exception as e2:
                print(f"Failed to load with direct method: {e2}")
                model = None
                print("Could not load the model with any method")
    except Exception as e:
        print(f"Error during model loading process: {e}")
        model = None

# Define image transformations for the model
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Classes for prediction
CLASSES = ["Normal", "Pneumonia"]

async def predict_xray(file):
    """
    Process the uploaded X-ray image and return prediction
    """
    if model is None:
        return {"error": "Model not loaded properly"}
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get the prediction class and confidence
        predicted_class = CLASSES[predicted.item()]
        confidence = probabilities[predicted.item()].item() * 100
        
        # Return the prediction results
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%",
            "probabilities": {
                CLASSES[i]: f"{prob.item() * 100:.2f}%" 
                for i, prob in enumerate(probabilities)
            }
        }
    except Exception as e:
        return {"error": str(e)}
