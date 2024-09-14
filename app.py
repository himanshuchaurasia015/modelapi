
from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('plant.h5')

# Define the folder to store uploaded images temporarily
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for image files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(225, 225)):
    """Preprocess the image for model prediction."""
    try:
        img = load_img(image_path, target_size=target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = preprocess_input(img)  # Use preprocess_input for the model
        return img
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

class ImagePredictor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def transform(self, X):
        results = []
        for img_path in X:
            img = image.load_img(img_path, target_size=(255, 255))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # Adjust based on your model's preprocessing

            # Make a prediction
            predictions = self.model.predict(img_array)
            res = np.argmax(predictions)

            # Return "Healthy" or "Not Healthy"
            results.append("Healthy" if res == 4 else "Not Healthy")
        return results
        # print(type(res))
        # return int(res)
        # return res

    def fit(self, X, y=None):
        return self  

# Create the pipeline
pipe = Pipeline(steps=[
    ("predictor", ImagePredictor(model))
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains files
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')

    # Ensure at least one file is provided
    if not files:
        return jsonify({"error": "No files provided"}), 400

    try:
        # Process each file and make predictions
        filepaths = []
        for file in files:
            if allowed_file(file.filename):
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                filepaths.append(filepath)
            else:
                return jsonify({"error": "Invalid file type"}), 400

        # Use the pipeline to make predictions
        results = pipe.transform(filepaths)

        # Calculate the percentage of healthy crops
        healthy_count = results.count("Healthy")
        total_count = len(results)
        healthy_percentage = (healthy_count / total_count) * 100

        response = {
            "healthy_percentage": healthy_percentage,
            "Unhealthy_percentage" : 100 - healthy_percentage
            }

    except Exception as e:
        response = {"error": str(e)}
    finally:
        # Remove the files after prediction
        for filepath in filepaths:
            os.remove(filepath)

    return jsonify(response), 200


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5002)
