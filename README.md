# 🧠 Brain Tumor Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![AI Framework: PyTorch](https://img.shields.io/badge/AI-PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

An AI-powered, full-stack web application for classifying brain tumors from MRI scans using a custom **ResNet-18** architecture. The system provides real-time clinical predictions, top-2 probability mapping, and explainability via **Grad-CAM heatmaps** to assist in research and educational diagnostic exploration. 

The system analyzes brain scans into four distinct categories: `no_tumor`, `glioma`, `meningioma`, and `pituitary`.

---

## ✨ Features

* **Real-Time Classification:** Upload MRI images and get instant classification predictions with precise confidence scores.
* **Explainable AI (XAI):** Integrated Grad-CAM visualizations to highlight exactly where the model is looking on the scan.
* **Top-2 Predictions:** Displays both primary and secondary tumor type possibilities with explicit probabilities.
* **Robust Image Validation:** Built-in backend guardrails to check file size (<10MB), image format, and dimensions.
* **Modern UI:** Responsive, dark-themed dashboard leveraging glassmorphism principles, smooth animations, and interactive drag-and-drop file inputs.
* **Performance Metrics:** Real-time visibility into CPU/GPU processing latency metrics per image scan.

---

## 🛠 Tech Stack

### Backend
* **Python 3.8+** - Core language environment.
* **FastAPI** - High-performance, low-latency web framework for building microservices.
* **PyTorch & Torchvision** - Deep learning framework utilized for model inference and preprocessing transformations.
* **OpenCV & Pillow** - Advanced image manipulation and Grad-CAM layer overlay blending.
* **Uvicorn** - Lightning-fast ASGI server implementation.

### Frontend
* **HTML5 & CSS3** - Modern structural semantic layout featuring glassmorphic styles (`backdrop-filter`).
* **JavaScript (ES6)** - Asynchronous client-side interactivity using the Native Fetch API.
* **Design Enhancements** - Google Fonts (Poppins, Montserrat) and Font Awesome icon configurations.

---

## 🏗 System Architecture

The application implements a clean, decoupled client-server pattern optimized for rapid local inference.

```text
[User Browser] ---> [Frontend (HTML/CSS/JS)]
                           |
                           v  (POST /classify API Request)
                  [FastAPI Server (backend/)]
                           |
         +-----------------+-----------------+
         |                 |                 |
         v                 v                 v
   [model.py]       [gradcam_lib.py]  [utils_pytorch.py]
  (ResNet-18)      (Heatmap Generation) (Image Validation)