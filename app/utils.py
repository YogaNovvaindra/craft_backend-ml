import os
import cv2
from flask import jsonify
import numpy as np
import importlib.util
from werkzeug.utils import secure_filename

from app.config import MODEL_DIR, GRAPH_NAME, LABELMAP_NAME, MIN_CONF_THRESHOLD, UPLOAD_FOLDER

# Load the label map
def load_labels():
    with open(os.path.join(MODEL_DIR, LABELMAP_NAME), 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    return labels

labels = load_labels()

# Load the TensorFlow Lite model
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path=os.path.join(MODEL_DIR, GRAPH_NAME))
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

def predict(request):
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No image provided"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(filepath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]  # Bounding box coordinates
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence

    detections = []
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            detections.append({
                "object": object_name,
                "confidence": float(scores[i]),
                "box": [xmin, ymin, xmax, ymax]
            })

    os.remove(filepath)
    return jsonify({"detections": detections})

def get_labels():
    return jsonify({"labels": labels})