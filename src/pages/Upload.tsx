
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Loader2, Upload, X, AlertCircle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

type UploadStatus = "idle" | "uploading" | "success" | "error";
interface PredictionResponse {
  filename: string;
  prediction: "Pneumonia" | "Normal";
  confidence: string;
  probabilities: {
    Normal: string;
    Pneumonia: string;
  };
}

type PredictionResult = null | PredictionResponse;

export default function UploadXray() {
  const [image, setImage] = useState<string | null>(null);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>("idle");
  const [predictionResult, setPredictionResult] = useState<PredictionResult>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    // Reset states
    setErrorMessage("");
    setPredictionResult(null);
    
    // Check if file is an image
    if (!file.type.match('image.*')) {
      setErrorMessage("Please upload an image file");
      return;
    }
    
    // Read and set the image
    const reader = new FileReader();
    reader.onload = () => {
      setImage(reader.result as string);
    };
    reader.readAsDataURL(file);
    
  // Make API call to FastAPI server
  setUploadStatus("uploading");
  
  // Create form data to send the file
  const formData = new FormData();
  formData.append('file', file);
  
  // Send the request to the FastAPI server
  fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData,
  })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      if (data.error) {
        throw new Error(data.error);
      }
      setPredictionResult(data);
      setUploadStatus("success");
    })
    .catch(error => {
      console.error("Error during prediction:", error);
      setUploadStatus("error");
      setErrorMessage(error.message || "Failed to analyze the image. Please try again.");
    });
  };


  const resetUpload = () => {
    setImage(null);
    setUploadStatus("idle");
    setPredictionResult(null);
    setErrorMessage("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="container max-w-5xl px-4 py-12 md:py-16 lg:py-24">
      <div className="flex flex-col items-center justify-center space-y-6 text-center">
        <h1 className="text-3xl font-bold tracking-tighter sm:text-4xl md:text-5xl">
          Upload Chest X-ray
        </h1>
        <p className="max-w-[700px] text-gray-500 md:text-xl dark:text-gray-400">
          Upload a chest X-ray image and our AI model will analyze it for signs of pneumonia.
        </p>
      </div>

      <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        <div className="flex flex-col space-y-6">
          <Card
            className={cn(
              "w-full border-2 border-dashed rounded-lg min-h-[300px] flex flex-col items-center justify-center p-6 transition-all",
              uploadStatus === "idle" && "hover:border-primary cursor-pointer"
            )}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => uploadStatus === "idle" && fileInputRef.current?.click()}
          >
            {uploadStatus === "idle" && (
              <div className="flex flex-col items-center justify-center space-y-4 text-center">
                <div className="rounded-full p-4 bg-secondary">
                  <Upload className="h-8 w-8 text-medical-600" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-medium text-xl">Upload X-ray Image</h3>
                  <p className="text-sm text-muted-foreground">
                    Drag and drop or click to upload a chest X-ray image
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Supports: JPG, PNG, JPEG
                  </p>
                </div>
              </div>
            )}

            {uploadStatus === "uploading" && (
              <div className="flex flex-col items-center justify-center space-y-4">
                <Loader2 className="h-12 w-12 text-primary animate-spin" />
                <div className="space-y-2">
                  <h3 className="font-medium text-xl">Analyzing X-ray</h3>
                  <p className="text-sm text-muted-foreground">
                    Please wait while our AI analyzes your image...
                  </p>
                </div>
              </div>
            )}

            {uploadStatus === "success" && (
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="rounded-full p-4 bg-green-100 dark:bg-green-900/20">
                  <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-500" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-medium text-xl">Analysis Complete</h3>
                  <p className="text-sm text-muted-foreground">
                    Your X-ray has been successfully analyzed
                  </p>
                </div>
              </div>
            )}

            {uploadStatus === "error" && (
              <div className="flex flex-col items-center justify-center space-y-4">
                <div className="rounded-full p-4 bg-red-100 dark:bg-red-900/20">
                  <AlertCircle className="h-8 w-8 text-red-600 dark:text-red-500" />
                </div>
                <div className="space-y-2">
                  <h3 className="font-medium text-xl">Error</h3>
                  <p className="text-sm text-muted-foreground">
                    {errorMessage || "Something went wrong. Please try again."}
                  </p>
                </div>
              </div>
            )}

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInput}
              accept="image/png, image/jpeg, image/jpg"
              className="hidden"
              aria-label="Upload X-ray image"
            />
          </Card>

          {uploadStatus !== "idle" && (
            <Button variant="outline" onClick={resetUpload} className="w-full">
              <X className="mr-2 h-4 w-4" />
              Upload New Image
            </Button>
          )}

          {errorMessage && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{errorMessage}</AlertDescription>
            </Alert>
          )}
        </div>

        <div className="flex flex-col space-y-6">
          <Card className="w-full border rounded-lg min-h-[300px] flex flex-col items-center justify-center p-6 overflow-hidden">
            {image ? (
              <div className="relative w-full h-full">
                <img
                  src={image}
                  alt="Uploaded X-ray"
                  className="w-full h-full object-contain"
                />
              </div>
            ) : (
              <div className="text-center text-muted-foreground flex flex-col items-center justify-center h-full">
                <p>X-ray preview will appear here</p>
              </div>
            )}
          </Card>

          {predictionResult && (
            <Card className={cn(
              "w-full border rounded-lg p-6",
              predictionResult.prediction === "Pneumonia" ? "bg-red-50 dark:bg-red-900/10 border-red-200 dark:border-red-800" : "bg-green-50 dark:bg-green-900/10 border-green-200 dark:border-green-800"
            )}>
              <div className="flex flex-col items-center justify-center space-y-4 text-center">
                <div className={cn(
                  "rounded-full p-4",
                  predictionResult.prediction === "Pneumonia" ? "bg-red-100 dark:bg-red-900/20" : "bg-green-100 dark:bg-green-900/20"
                )}>
                  {predictionResult.prediction === "Pneumonia" ? (
                    <AlertCircle className="h-8 w-8 text-red-600 dark:text-red-500" />
                  ) : (
                    <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-500" />
                  )}
                </div>
                <div className="space-y-2">
                  <h3 className="font-medium text-xl">Prediction Result</h3>
              <p className={cn(
                "text-2xl font-bold",
                predictionResult.prediction === "Pneumonia" ? "text-red-600 dark:text-red-500" : "text-green-600 dark:text-green-500"
              )}>
                {predictionResult.prediction}
              </p>
              <p className="text-lg font-medium mt-1">
                Confidence: {predictionResult.confidence}
              </p>
              <div className="w-full mt-3 space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Normal:</span>
                  <span>{predictionResult.probabilities.Normal}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Pneumonia:</span>
                  <span>{predictionResult.probabilities.Pneumonia}</span>
                </div>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                {predictionResult.prediction === "Pneumonia" 
                  ? "The X-ray shows signs consistent with pneumonia. Please consult with a healthcare professional." 
                  : "The X-ray appears normal with no significant signs of pneumonia."}
              </p>
                </div>
              </div>
            </Card>
          )}

          {/* Additional information */}
          {!predictionResult && (
            <Card className="w-full border rounded-lg p-6">
              <div className="space-y-4">
                <h3 className="font-medium text-xl">How It Works</h3>
                <p className="text-sm text-muted-foreground">
                  Our AI model analyzes chest X-rays to detect patterns associated with pneumonia. 
                  The analysis is performed in real-time and results are provided within seconds.
                </p>
                <div className="pt-2">
                  <h4 className="font-medium text-sm mb-2">For best results:</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Upload a clear, high-resolution X-ray image</li>
                    <li>• Ensure the image shows the full chest area</li>
                    <li>• Use PA (posteroanterior) or AP (anteroposterior) views</li>
                  </ul>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
