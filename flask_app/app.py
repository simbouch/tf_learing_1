import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

# Create Flask app
app = Flask(__name__)

# Load the saved Keras model
MODEL_PATH = os.path.join("..", "models", "beans_cnn_model.keras")  # Update path if needed
loaded_model = tf.keras.models.load_model(MODEL_PATH)

# Ensure model is loaded correctly
print(f"Model loaded successfully from {MODEL_PATH}")
print(f"Model input shape: {loaded_model.input_shape}")

# Class labels mapping (in French)
CLASS_LABELS = {
    0: "Tâche angulaire sur feuille (Angular Leaf Spot): Une maladie fongique qui cause des taches sur les feuilles.",
    1: "Rouille du haricot (Bean Rust): Une maladie causée par des champignons rouille, apparaissant souvent sous forme de pustules.",
    2: "Feuille saine (Healthy): La feuille semble saine et exempte de maladies.",
}

# Preprocessing function for images
def preprocess_image(image, target_size):
    """
    Preprocess the uploaded image to match the model input requirements.
    """
    # Resize the image to the target size
    image = image.resize(target_size)
    # Convert image to numpy array
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # Ensure the image has the correct shape (add batch dimension)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def home():
    """
    Render the homepage with upload instructions.
    """
    return render_template("index.html")  # Create an `index.html` file in your `templates` directory.

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class of the uploaded bean image.
    """
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400

        # Get the uploaded file
        file = request.files['file']

        # Open the image file
        img = Image.open(file.stream).convert('RGB')  # Ensure 3-channel RGB image

        # Save uploaded image for display
        img_path = os.path.join("static", "uploaded_image.jpg")
        img.save(img_path)

        # Preprocess the image
        processed_img = preprocess_image(img, target_size=(500, 500))  # Match the model input size

        # Predict using the loaded model
        predictions = loaded_model.predict(processed_img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Get the descriptive label
        result_label = CLASS_LABELS.get(predicted_class, "Inconnu")

        # Render the result page with the uploaded image and prediction
        return render_template(
            "result.html",
            image_url=img_path,
            prediction=result_label
        )

    except Exception as e:
        # Handle errors gracefully
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
