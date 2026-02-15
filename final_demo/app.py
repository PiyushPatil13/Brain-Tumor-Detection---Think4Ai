from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import cv2
from typing import Dict, Tuple
import time

# Import custom modules
from gradcam_lib import apply_gradcam_library
from model import BrainTumorModel
from utils_pytorch import (
    preprocess_for_model,
    load_image_from_bytes,
    validate_image,
    format_prediction_response,
    create_error_response,
    get_class_names
)

app = FastAPI(title="Brain Tumor Classification API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global model instance
model_instance = None

@app.on_event("startup")
async def startup_event():
    global model_instance
    model_path = "best_brain_tumor_cnn.pth"  # Update if path differs
    try:
        model_instance = BrainTumorModel(model_path)
        print("Model loaded at startup.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def generate_gradcam(model: BrainTumorModel, image_tensor: torch.Tensor, original_array: np.ndarray, predicted_class: int) -> np.ndarray:
    """Generate GradCAM heatmap."""
    target_layer = model.get_target_layer()
    heatmap, overlay = apply_gradcam_library(
        model=model.get_model(),
        image_tensor=image_tensor,
        original_image=original_array,
        target_layer=target_layer,
        target_class=predicted_class
    )
    return overlay

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        validation = validate_image(image_bytes)
        if not validation["valid"]:
            raise HTTPException(status_code=400, detail=validation["message"])
        
        pil_image, original_array = load_image_from_bytes(image_bytes)
        image_tensor = preprocess_for_model(pil_image)
        
        start_time = time.time()
        prediction = model_instance.predict(image_tensor)
        processing_time = time.time() - start_time
        
        predicted_class_idx = get_class_names().index(prediction["predicted_class"])
        gradcam_image = generate_gradcam(model_instance, image_tensor, original_array, predicted_class_idx)
        
        response = format_prediction_response(prediction, gradcam_image, processing_time)
        return response
    
    except Exception as e:
        return create_error_response(str(e))

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brain Tumor Classifier - AI-Powered Diagnosis</title>
        <link rel="icon" href="logo.png" type="image/png">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700&family=Montserrat:wght@700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: 'Poppins', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
                color: #e0e0e0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
                position: relative;
            }
            body::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at 20% 80%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
                            radial-gradient(circle at 80% 20%, rgba(138, 43, 226, 0.1) 0%, transparent 50%);
                pointer-events: none;
                z-index: -1;
            }
            .header {
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(10px);
                padding: 30px 20px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(0, 255, 255, 0.2);
                border-bottom: 1px solid rgba(0, 255, 255, 0.3);
            }
            .header h1 {
                margin: 0;
                color: #00ffff;
                font-family: 'Montserrat', sans-serif;
                font-size: 3em;
                font-weight: 700;
                text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
                animation: type 3s steps(30, end);
                overflow: hidden;
                white-space: nowrap;
                border-right: 2px solid #00ffff;
            }
            @keyframes type {
                from { width: 0; }
                to { width: 100%; }
            }
            .header p {
                margin: 15px 0 0;
                color: #b0b0b0;
                font-size: 1.2em;
                opacity: 0;
                animation: fadeIn 2s 1s forwards;
            }
            @keyframes fadeIn {
                to { opacity: 1; }
            }
            .container {
                max-width: 1000px;
                margin: 30px auto;
                padding: 30px;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(15px);
                border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.1);
                flex-grow: 1;
                animation: slideUp 0.8s ease-out;
            }
            @keyframes slideUp {
                from { transform: translateY(50px); opacity: 0; }
                to { transform: translateY(0); opacity: 1; }
            }
            .upload-section {
                text-align: center;
                margin-bottom: 40px;
            }
            .upload-section h2 {
                color: #00ffff;
                margin-bottom: 20px;
                font-size: 1.8em;
                text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
            }
            .file-input {
                display: inline-block;
                margin-bottom: 20px;
                padding: 20px;
                border: 2px dashed #00ffff;
                border-radius: 15px;
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                cursor: pointer;
                transition: all 0.4s;
                color: #e0e0e0;
                font-size: 1.1em;
            }
            .file-input:hover {
                border-color: #8a2be2;
                box-shadow: 0 0 20px rgba(138, 43, 226, 0.5);
                transform: scale(1.05);
            }
            .preview {
                max-width: 250px;
                max-height: 250px;
                margin: 20px auto;
                border-radius: 15px;
                display: none;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
                border: 2px solid rgba(0, 255, 255, 0.3);
            }
            button {
                background: linear-gradient(135deg, #00ffff, #8a2be2);
                color: #000;
                border: none;
                padding: 15px 30px;
                border-radius: 12px;
                font-size: 1.1em;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.4s;
                margin: 10px;
                box-shadow: 0 4px 15px rgba(0, 255, 255, 0.3);
            }
            button:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0, 255, 255, 0.6);
            }
            button:disabled {
                background: #333;
                cursor: not-allowed;
                box-shadow: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 30px 0;
            }
            .spinner {
                border: 5px solid rgba(255, 255, 255, 0.1);
                border-top: 5px solid #00ffff;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
                box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                display: none;
                animation: fadeInUp 0.6s;
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(20px);
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 255, 255, 0.2);
            }
            .card h3 {
                color: #00ffff;
                margin-top: 0;
                font-size: 1.4em;
                display: flex;
                align-items: center;
            }
            .card h3 i {
                margin-right: 10px;
            }
            .prediction {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .confidence-bar {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                height: 10px;
                width: 100%;
                margin-left: 15px;
                overflow: hidden;
            }
            .confidence-fill {
                background: linear-gradient(90deg, #00ffff, #8a2be2);
                height: 100%;
                border-radius: 6px;
                transition: width 1s ease;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 12px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
            }
            .error {
                background: rgba(255, 0, 0, 0.1);
                color: #ff6b6b;
                padding: 20px;
                border-radius: 12px;
                display: none;
                text-align: center;
                border: 1px solid rgba(255, 0, 0, 0.3);
                backdrop-filter: blur(10px);
            }
            .footer {
                text-align: center;
                padding: 25px;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(10px);
                color: #b0b0b0;
                font-size: 0.9em;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            .footer strong {
                color: #ff6b6b;
            }
            @media (max-width: 768px) {
                .container { margin: 15px; padding: 20px; }
                .header h1 { font-size: 2.2em; }
                .prediction { flex-direction: column; align-items: flex-start; }
                .card { padding: 20px; }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1><i class="fas fa-brain"></i> Brain Tumor Classifier</h1>
            <p>AI-Powered MRI Analysis for Research and Education</p>
        </div>
        <div class="container">
            <div class="upload-section">
                <h2><i class="fas fa-upload"></i> Upload Brain MRI Scan</h2>
                <input type="file" id="imageInput" accept="image/*" style="display: none;">
                <label for="imageInput" class="file-input"><i class="fas fa-file-image"></i> Choose Image or Drag & Drop</label>
                <br>
                <img id="imagePreview" class="preview" alt="Image Preview">
                <br>
                <button id="classifyBtn" onclick="uploadImage()"><i class="fas fa-search"></i> Analyze Scan</button>
                <button onclick="resetForm()"><i class="fas fa-redo"></i> Reset</button>
            </div>
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analyzing image... Please wait.</p>
            </div>
            <div id="results" class="results">
                <div class="card">
                    <h3><i class="fas fa-star"></i> Primary Prediction</h3>
                    <div class="prediction">
                        <span id="primary"></span>
                        <div class="confidence-bar"><div class="confidence-fill" id="primaryBar" style="width: 0%;"></div></div>
                    </div>
                </div>
                <div class="card">
                    <h3><i class="fas fa-chart-line"></i> Secondary Prediction</h3>
                    <div class="prediction">
                        <span id="secondary"></span>
                        <div class="confidence-bar"><div class="confidence-fill" id="secondaryBar" style="width: 0%;"></div></div>
                    </div>
                </div>
                <div class="card">
                    <h3><i class="fas fa-list"></i> All Predictions</h3>
                    <ul id="allPredictions"></ul>
                </div>
                <div class="card">
                    <h3><i class="fas fa-eye"></i> GradCAM Visualization</h3>
                    <p>Heatmap showing model focus areas</p>
                    <img id="gradcamImage" alt="GradCAM Heatmap">
                </div>
                <div class="card">
                    <p id="processingTime"></p>
                </div>
            </div>
            <div id="error" class="error">
                <i class="fas fa-exclamation-triangle"></i> <strong>Error:</strong> <span id="errorMsg"></span>
            </div>
        </div>
        <div class="footer">
            <p><i class="fas fa-info-circle"></i> <strong>Disclaimer:</strong> This tool is for research purposes only. Not a substitute for professional medical diagnosis. Consult a qualified radiologist.</p>
        </div>
        <script>
            const imageInput = document.getElementById('imageInput');
            const imagePreview = document.getElementById('imagePreview');
            const classifyBtn = document.getElementById('classifyBtn');
            
            // Image preview
            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            // Drag & drop
            document.addEventListener('dragover', (e) => e.preventDefault());
            document.addEventListener('drop', (e) => {
                e.preventDefault();
                const file = e.dataTransfer.files[0];
                if (file) {
                    imageInput.files = e.dataTransfer.files;
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        imagePreview.src = e.target.result;
                        imagePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            async function uploadImage() {
                const file = imageInput.files[0];
                if (!file) {
                    alert('Please select an image first.');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                classifyBtn.disabled = true;
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/classify', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('primary').textContent = `${data.prediction.class} (${data.prediction.confidence_percentage})`;
                        document.getElementById('primaryBar').style.width = `${data.prediction.confidence}%`;
                        document.getElementById('secondary').textContent = `${data.secondary_prediction.class} (${data.secondary_prediction.confidence_percentage})`;
                        document.getElementById('secondaryBar').style.width = `${data.secondary_prediction.confidence}%`;
                        document.getElementById('allPredictions').innerHTML = data.all_predictions.map(p => `<li>${p.class}: ${p.percentage}</li>`).join('');
                        document.getElementById('gradcamImage').src = data.gradcam_image;
                        document.getElementById('processingTime').textContent = `Processing time: ${data.processing_time.milliseconds} ms`;
                        document.getElementById('results').style.display = 'block';
                    } else {
                        document.getElementById('errorMsg').textContent = data.error.message;
                        document.getElementById('error').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('errorMsg').textContent = 'Network error. Please try again.';
                    document.getElementById('error').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    classifyBtn.disabled = false;
                }
            }
            
            function resetForm() {
                imageInput.value = '';
                imagePreview.style.display = 'none';
                document.getElementById('results').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                document.getElementById('loading').style.display = 'none';
                classifyBtn.disabled = false;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)