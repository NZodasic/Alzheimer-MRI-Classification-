import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model = tf.keras.models.load_model("model\model.h5")

# Define class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image = image.resize((244, 244))
        image_array = np.array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        return {"predicted_label": predicted_label}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8000)