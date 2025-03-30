# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


def analyze_leaf(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Unable to read image."
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_leaf = cv2.inRange(hsv, lower_green, upper_green)
    leaf_area = np.count_nonzero(mask_leaf)
    
    lower_brown = np.array([10, 50, 50])
    upper_brown = np.array([30, 255, 255])
    mask_lesion = cv2.inRange(hsv, lower_brown, upper_brown)
    lesion_area = np.count_nonzero(mask_lesion)
    
    if leaf_area == 0:
        return None, "Error: No leaf detected."
    
    severity = (lesion_area / leaf_area) * 100
    result_text = f"Total Leaf Area: {leaf_area} pixels²\nLesion Area: {lesion_area} pixels²\nDisease Severity: {severity:.2f}%"
    
    contours, _ = cv2.findContours(mask_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    result_path = os.path.join(RESULT_FOLDER, os.path.basename(image_path))
    cv2.imwrite(result_path, image)
    
    return os.path.basename(result_path), result_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    result_image, analysis_text = analyze_leaf(file_path)
    if result_image is None:
        return jsonify({"error": analysis_text})
    
    return jsonify({"image": url_for('static', filename=f'results/{result_image}'), "result": analysis_text})


if __name__ == '__main__':
    app.run(debug=True)