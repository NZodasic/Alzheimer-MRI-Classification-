import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
try:
    model = tf.keras.models.load_model("model/model.h5")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

app = FastAPI(title="Alzheimer's Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_image(image):
    """Validate image dimensions and format"""
    if image.format not in ['JPEG', 'PNG']:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")
    
    width, height = image.size
    if width < 50 or height < 50:
        raise HTTPException(status_code=400, detail="Image dimensions too small")
    
    return True

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict Alzheimer's stage from brain MRI image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        validate_image(image)
        
        # Preprocess image
        image = image.convert('RGB')  # Ensure RGB format
        image = image.resize((244, 244))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]
        confidence = float(prediction[0][predicted_class])  # Convert to float for JSON serialization
        
        logger.info(f"Prediction made: {predicted_label} with confidence {confidence:.2f}")
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "success": True
        }
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)