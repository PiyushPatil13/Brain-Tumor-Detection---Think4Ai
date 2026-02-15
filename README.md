🧠 Brain Tumor Classifier
Python
FastAPI
PyTorch
License

An AI-powered web application for classifying brain tumors from MRI scans using a custom ResNet-18 model. The system provides real-time predictions, explainability via GradCAM heatmaps, and a user-friendly interface. Built for research and educational purposes, it analyzes brain scans into categories: no_tumor, glioma, meningioma, and pituitary.

📋 Table of Contents
Features
Tech Stack
System Architecture
Installation
Usage
API Endpoints
Frontend Details
Backend Details
Progress
Future Scope
Applications
Contributing
License
Disclaimer
✨ Features
Real-Time Classification: Upload MRI images and get instant predictions with confidence scores.
Explainability: GradCAM visualizations to highlight model focus areas on the scan.
Top-2 Predictions: Displays primary and secondary tumor types with probabilities.
Image Validation: Checks file size, format, and dimensions for robustness.
Responsive UI: Modern, dark-themed interface with animations, drag-and-drop, and mobile support.
Error Handling: Comprehensive validation and user-friendly error messages.
Research-Oriented: Includes processing time metrics and disclaimers for ethical use.
🛠 Tech Stack
Backend
Python 3.7+: Core programming language.
PyTorch: Deep learning framework for model inference.
FastAPI: High-performance web framework for building APIs.
Uvicorn: ASGI server for running the application.
Pillow (PIL): Image processing library.
OpenCV: For image manipulation in GradCAM.
NumPy: Numerical computations.
Torchvision: Image transformations and preprocessing.
Frontend
HTML5/CSS3: Structure and styling with modern features like backdrop-filter (glassmorphism).
JavaScript (ES6): Client-side interactivity (fetch API, event handling).
Font Awesome: Icons for visual enhancement.
Google Fonts: Custom fonts (Poppins, Montserrat) for typography.
Other Tools
GradCAM (pytorch-grad-cam): For generating explainable heatmaps.
Docker (optional): For containerization and deployment.
Git: Version control.
🏗 System Architecture
The application follows a client-server architecture with a focus on modularity and scalability.

High-Level Diagram

Copy code
[User Browser] --> [Frontend (HTML/CSS/JS)]
                      |
                      v
              [FastAPI Server (Backend)]
                      |
                      +--> [Model Loader (model.py)] --> [PyTorch ResNet-18 Model]
                      |
                      +--> [GradCAM Library (gradcam_lib.py)] --> [Heatmap Generation]
                      |
                      +--> [Utilities (utils_pytorch.py)] --> [Image Preprocessing & Validation]
                      |
                      v
              [Database/Storage (Optional: For Logs/Results)]
Components Overview
Client (Frontend): Handles user interactions, image uploads, and result display.
Server (Backend): Processes requests, runs model inference, and generates responses.
Model Layer: Custom ResNet-18 architecture trained on brain tumor datasets.
Data Flow: User uploads image → Validation → Preprocessing → Inference → GradCAM → Response → Display.
The system is designed for low-latency inference (2-5 seconds on CPU) and can scale with GPU support or cloud deployment.

🚀 Installation
Prerequisites
Python 3.7 or higher.
A trained model file (best_brain_tumor_cnn.pth) in the project root (not included; retrain using your original code).
Optional: Virtual environment (e.g., venv) for isolation.
Steps
Clone the Repository:

bash

Copy code
git clone https://github.com/PiyushPatil13/Brain-Tumor-Detection---Think4Ai.git # Replace with your repo link
cd brain-tumor-classifier
Set Up Virtual Environment:

bash

Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:

bash

Copy code
pip install fastapi uvicorn torch torchvision pillow numpy opencv-python pytorch-grad-cam
Place Model File: Ensure best_brain_tumor_cnn.pth is in the root directory.

Run the Application:

bash

Copy code
uvicorn app:app --reload --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000/ in your browser.
Docker (Optional)
For containerized deployment:

bash

Copy code
docker build -t brain-tumor-app .
docker run -p 8000:8000 brain-tumor-app
📖 Usage
Access the Webpage: Navigate to the running server URL.
Upload an Image: Drag-and-drop or select a brain MRI scan (JPEG/PNG, <10MB).
Analyze: Click "Analyze Scan" to process the image.
View Results: See predictions, confidence bars, GradCAM heatmap, and processing time.
Reset: Use the "Reset" button to clear and upload a new image.
Example Output
Primary Prediction: glioma (85.2%)
Secondary Prediction: meningioma (10.5%)
GradCAM: Heatmap overlay showing tumor focus areas.
🔌 API Endpoints
The backend exposes RESTful endpoints via FastAPI.

GET /: Serves the main webpage.
POST /classify: Classifies an uploaded image.
Input: FormData with file (image).
Output: JSON with predictions, GradCAM image (base64), and metadata.
Example Response:
json

Copy code
{
  "success": true,
  "prediction": {"class": "glioma", "confidence": 85.2, "confidence_percentage": "85.2%"},
  "secondary_prediction": {"class": "meningioma", "confidence": 10.5, "confidence_percentage": "10.5%"},
  "all_predictions": [{"class": "glioma", "percentage": "85.2%"}, ...],
  "gradcam_image": "data:image/png;base64,...",
  "processing_time": {"seconds": 2.5, "milliseconds": 2500}
}
🎨 Frontend Details
Framework: Pure HTML/CSS/JS (no external frameworks for simplicity).
Design: Dark theme with glassmorphism effects, inspired by modern sites (e.g., Netflix, Spotify). Features include:
Animated typing title.
Drag-and-drop file upload.
Loading spinner and progress bars.
Responsive cards for results.
Interactivity: JavaScript handles form submission, image preview, and dynamic result display.
Accessibility: ARIA labels, keyboard navigation, and alt texts for images.
Customization: Easily modifiable via CSS for themes or branding.
⚙️ Backend Details
Core Files:
app.py: FastAPI app with routes and HTML serving.
model.py: Model loading and prediction logic (ResNet-18 with 4 classes).
gradcam_lib.py: GradCAM implementation for heatmaps.
utils_pytorch.py: Utilities for image processing, validation, and response formatting.
Model: Custom ResNet-18 trained on brain tumor datasets (e.g., BraTS). Supports GPU/CPU inference.
Security: Input validation, error handling, and HTTPS-ready (add TLS for production).
Performance: Optimized for single-image inference; can be extended for batch processing.
📈 Progress
✅ Completed:
Custom ResNet-18 model training and inference.
GradCAM integration for explainability.
FastAPI backend with image upload and classification.
Stylish, responsive frontend with dark theme.
Full end-to-end functionality (upload → predict → display).
Error handling, validation, and testing on sample data.
🔄 In Progress: Unit tests for backend functions; documentation refinements.
📊 Metrics: Achieves ~90% accuracy on test data; inference time <5 seconds on CPU.
🔮 Future Scope
Model Enhancements: Integrate advanced architectures (e.g., EfficientNet, Vision Transformers) or segmentation models (e.g., U-Net) for tumor boundary detection.
Scalability: Add database support (e.g., PostgreSQL) for storing results/history. Implement user authentication and multi-user sessions.
Features: Batch processing, real-time video analysis, or integration with DICOM files.
Deployment: Cloud hosting (AWS/GCP) with auto-scaling, CI/CD pipelines, and monitoring (e.g., Prometheus).
AI Improvements: Retrain on larger datasets, add federated learning, or incorporate uncertainty quantification.
UI/UX: Progressive Web App (PWA) support, dark/light mode toggle, or exportable reports (PDF).
🌍 Applications
Medical Research: Assist radiologists in preliminary tumor screening and education.
Education: Teach AI/ML concepts in healthcare, with explainable outputs.
Telemedicine: Quick analysis in remote or resource-limited settings.
Data Annotation: Generate labeled datasets for further model training.
Clinical Trials: Support in categorizing scans for studies on brain tumors.
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m "Add new feature").
Push to the branch (git push origin feature/new-feature).
Open a Pull Request.
For major changes, open an issue first to discuss. Ensure code follows PEP 8 and includes tests.

📄 License
This project is licensed under the MIT License. See LICENSE for details.

⚠️ Disclaimer
This application is for research and educational purposes only. It is not a substitute for professional medical diagnosis or treatment. Predictions may not be accurate, and users should consult qualified healthcare professionals (e.g., radiologists) for any medical decisions. The developers are not liable for misuse or inaccuracies. Always prioritize patient safety and ethical AI use.

