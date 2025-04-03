import os
from flask import Blueprint, request, jsonify, make_response
from werkzeug.utils import secure_filename

from .model_loader import model
from .utils import preprocess_image
from data.dataset_loader import load_data
from config.config import dataset_path
from flask_cors import cross_origin
from inference.predict import predict_single_image


# import os
import sys

# Tambahkan jalur root project secara manual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


routes = Blueprint("routes", __name__)

(X_train, X_test, y_train, y_test), label_encoder = load_data(dataset_path)


@routes.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin(origins=["http://localhost:8000"])
def predict_image():
    if request.method == 'OPTIONS':
        # Tangani preflight request
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:8000")
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join("api/upload", filename)
    image_file.save(filepath)

    predicted_class = predict_single_image(model, filepath, label_encoder=label_encoder)

    return jsonify({
        "filename": filename,
        "prediction": predicted_class
    })
