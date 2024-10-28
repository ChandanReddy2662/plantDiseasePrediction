from flask import Flask, render_template, request
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename
from io import BytesIO
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Load the trained model
model = tf.keras.models.load_model('C:/Users/saich/OneDrive/Desktop/djangoChatApp/plant_disease/plant_disease_model.h5')

# Define image size and class labels
image_size = (128, 128)
class_labels = {
    0: 'Pepper__bell___Bacterial_spot', 1: 'Pepper__bell___healthy', 
    2: 'Potato___Early_blight', 3: 'Potato___Late_blight', 4: 'Potato___healthy', 
    5: 'Tomato_Bacterial_spot', 6: 'Tomato_Early_blight', 7: 'Tomato_Late_blight', 
    8: 'Tomato_Leaf_Mold'
}
@app.route("/", methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    image_data = None  # To store the base64 string of the image

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        print(file)
        if file.filename == '':
            return "No selected file"

        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        image = load_img(file_path, target_size=image_size)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Prepare image for the model

        # Make prediction
        prediction_probs = model.predict(image_array)
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = class_labels[predicted_class_idx]
        probability = np.max(prediction_probs)  # Get the maximum probability

        # Convert the image to base64 for rendering in HTML
        buffered = io.BytesIO()
        # Save the original PIL image (not the numpy array) to the buffer
        image.save(buffered, format="JPEG")
        image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return render_template('index.html', prediction=predicted_class, probability=probability, image_data=image_data)

    return render_template('index.html', prediction=None, probability=None)

if __name__ == "__main__":
    app.run(debug=True)