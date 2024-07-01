from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = load_model('fer2013_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read the image from the request
        nparr = np.frombuffer(request.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400
        img = cv2.resize(img, (48, 48))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Make prediction
        prediction = model.predict(img)
        emotion_index = np.argmax(prediction)
        emotion = emotion_labels[emotion_index]

        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)