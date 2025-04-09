from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.model import predict_xray

app = FastAPI(title="X-Ray Insight API", description="API for X-Ray pneumonia prediction")

# Configure CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "X-Ray Insight API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an X-ray image and get pneumonia prediction
    """
    try:
        # Process the uploaded image and get prediction
        result = await predict_xray(file)
        return result
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
