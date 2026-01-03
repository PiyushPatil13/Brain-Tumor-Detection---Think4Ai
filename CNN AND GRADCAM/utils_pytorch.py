import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import base64
import cv2
from typing import Tuple, Dict, Union

# ==================== PYTORCH IMAGE PREPROCESSING ====================

# Define transforms (same as your training)
test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def preprocess_for_model(image: Image.Image) -> torch.Tensor:
    """
    Preprocess PIL Image for PyTorch model
    
    Args:
        image: PIL Image
    
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = test_transforms(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor


def load_image_from_bytes(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    """
    Load image from bytes
    
    Args:
        image_bytes: Image file bytes
    
    Returns:
        Tuple of (PIL Image, numpy array)
    """
    # Load PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array for GradCAM
    image_array = np.array(image)
    
    return image, image_array


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize tensor back to original image
    Used for visualization
    
    Args:
        tensor: Normalized tensor (C, H, W)
    
    Returns:
        Denormalized numpy array (H, W, C)
    """
    # ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    image = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert to 0-255
    image = (image * 255).astype(np.uint8)
    
    return image


# ==================== VALIDATION ====================

def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> Dict:
    """
    Validate uploaded image
    
    Args:
        image_bytes: Image bytes
        max_size_mb: Maximum size in MB
    
    Returns:
        Validation result dictionary
    """
    # Check file size
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return {
            "valid": False,
            "message": f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        }
    
    # Try to open image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check format
        if image.format not in ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF']:
            return {
                "valid": False,
                "message": f"Unsupported format: {image.format}"
            }
        
        # Check dimensions
        width, height = image.size
        if width < 50 or height < 50:
            return {
                "valid": False,
                "message": "Image too small (min 50x50 pixels)"
            }
        
        return {
            "valid": True,
            "message": "Valid image",
            "format": image.format,
            "size": (width, height)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"Invalid image: {str(e)}"
        }


# ==================== ENCODING/DECODING ====================

def numpy_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy array to base64 string
    
    Args:
        image_array: Image as numpy array
    
    Returns:
        Base64 encoded string
    """
    # Ensure uint8
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image
    image = Image.fromarray(image_array)
    
    # Save to buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy array
    
    Args:
        base64_string: Base64 encoded image
    
    Returns:
        Numpy array
    """
    # Remove data URL prefix
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Decode
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL and then numpy
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    return image_array


# ==================== RESPONSE FORMATTING ====================

def format_prediction_response(
    prediction: Dict,
    gradcam_image: np.ndarray = None,
    processing_time: float = None
) -> Dict:
    """
    Format prediction response for API
    
    Args:
        prediction: Prediction dictionary from model
        gradcam_image: GradCAM visualization (optional)
        processing_time: Processing time in seconds
    
    Returns:
        Formatted response
    """
    response = {
        "success": True,
        "prediction": {
            "class": prediction["predicted_class"],
            "confidence": round(prediction["confidence"], 2),
            "confidence_percentage": f"{prediction['confidence']:.2f}%"
        },
        "secondary_prediction": {
            "class": prediction["secondary_class"],
            "confidence": round(prediction["secondary_confidence"], 2),
            "confidence_percentage": f"{prediction['secondary_confidence']:.2f}%"
        },
        "all_predictions": prediction["all_predictions"]
    }
    
    # Add GradCAM if available
    if gradcam_image is not None:
        response["gradcam_image"] = numpy_to_base64(gradcam_image)
    
    # Add processing time
    if processing_time is not None:
        response["processing_time"] = {
            "seconds": round(processing_time, 3),
            "milliseconds": round(processing_time * 1000, 2)
        }
    
    return response


def create_error_response(error_message: str, error_code: str = "PROCESSING_ERROR") -> Dict:
    """
    Create error response
    
    Args:
        error_message: Error message
        error_code: Error code
    
    Returns:
        Error response dictionary
    """
    return {
        "success": False,
        "error": {
            "message": error_message,
            "code": error_code
        }
    }


# ==================== CLASS NAMES ====================

def get_class_names() -> list:
    """
    Get brain tumor class names
    Must match your training dataset classes
    
    Returns:
        List of class names
    """
    return ['glioma', 'meningioma', 'notumor', 'pituitary']


def get_class_info() -> Dict:
    """
    Get detailed class information
    
    Returns:
        Dictionary with class descriptions
    """
    return {
        "glioma": {
            "name": "Glioma",
            "description": "A tumor that starts in glial cells of the brain or spine",
            "severity": "high"
        },
        "meningioma": {
            "name": "Meningioma",
            "description": "A tumor that forms on membranes covering brain and spinal cord",
            "severity": "medium"
        },
        "notumor": {
            "name": "No Tumor",
            "description": "No tumor detected in the brain scan",
            "severity": "none"
        },
        "pituitary": {
            "name": "Pituitary Tumor",
            "description": "A tumor in the pituitary gland",
            "severity": "medium"
        }
    }