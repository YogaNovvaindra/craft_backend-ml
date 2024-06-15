import os

MODEL_DIR = "models"
GRAPH_NAME = "detect.tflite"
LABELMAP_NAME = "labelmap.txt"
MIN_CONF_THRESHOLD = 0.5
UPLOAD_FOLDER = "uploads"

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)