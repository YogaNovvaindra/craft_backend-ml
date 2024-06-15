from flask import request, jsonify
from app import app
from app.utils import predict, get_labels

@app.route('/predict', methods=['POST'])
def predict_route():
    return predict(request)

@app.route('/labels', methods=['GET'])
def labels_route():
    return get_labels()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "service is running"})