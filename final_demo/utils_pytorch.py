import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import base64
import cv2
from typing import Tuple, Dict, Union, List

# Define transforms (consistent with your training/test code)
test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_for_model(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model input.
    
    Args:
        image: PIL Image object.
    
    Returns:
        Preprocessed tensor ready for model inference.
    
    Raises:
        ValueError: If image conversion fails.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = test_transforms(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")

def load_image_from_bytes(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    """
    Load image from bytes into PIL Image and numpy array.
    
    Args:
        image_bytes: Raw image bytes.
    
    Returns:
        Tuple of (PIL Image, numpy array).
    
    Raises:
        ValueError: If loading fails.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes provided.")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        return image, image_array
    except Exception as e:
        raise ValueError(f"Failed to load image from bytes: {str(e)}")

def validate_image(image_bytes: bytes, max_size_mb: int = 10) -> Dict[str, Union[bool, str, Tuple[int, int]]]:
    """
    Validate uploaded image for size, format, and dimensions.
    
    Args:
        image_bytes: Raw image bytes.
        max_size_mb: Maximum allowed file size in MB.
    
    Returns:
        Dict with 'valid' (bool), 'message' (str), and optional 'format'/'size'.
    """
    if not image_bytes:
        return {
            "valid": False,
            "message": "No image data provided."
        }
    
    file_size_mb = len(image_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return {
            "valid": False,
            "message": f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        }
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        # Check format
        allowed_formats = ['JPEG', 'JPG', 'PNG', 'BMP', 'TIFF']
        if image.format not in allowed_formats:
            return {
                "valid": False,
                "message": f"Unsupported format: {image.format}. Allowed: {', '.join(allowed_formats)}"
            }
        
        # Check dimensions
        width, height = image.size
        min_dim = 50
        if width < min_dim or height < min_dim:
            return {
                "valid": False,
                "message": f"Image too small: {width}x{height} (min: {min_dim}x{min_dim} pixels)"
            }
        
        # Additional check: Ensure it's not corrupted
        image.verify()  # Raises exception if corrupted
        
        return {
            "valid": True,
            "message": "Valid image",
            "format": image.format,
            "size": (width, height)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": f"Invalid or corrupted image: {str(e)}"
        }

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor back to a displayable image array.
    
    Args:
        tensor: Normalized tensor (C, H, W).
    
    Returns:
        Denormalized numpy array (H, W, C) in uint8 format.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    image = tensor.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def numpy_to_base64(image_array: np.ndarray) -> str:
    """
    Convert numpy array to base64-encoded PNG string.
    
    Args:
        image_array: Numpy array (H, W, C).
    
    Returns:
        Base64 string prefixed with data URL.
    """
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def base64_to_numpy(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to numpy array.
    
    Args:
        base64_string: Base64 string (with or without data URL prefix).
    
    Returns:
        Numpy array (H, W, C).
    """
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    return image_array

def format_prediction_response(
    prediction: Dict,
    gradcam_image: np.ndarray = None,
    processing_time: float = None
) -> Dict:
    """
    Format prediction results into a standardized response.
    
    Args:
        prediction: Dict from model prediction.
        gradcam_image: Optional numpy array for GradCAM.
        processing_time: Optional processing time in seconds.
    
    Returns:
        Formatted response dict.
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
    
    if gradcam_image is not None:
        response["gradcam_image"] = numpy_to_base64(gradcam_image)
    
    if processing_time is not None:
        response["processing_time"] = {
            "seconds": round(processing_time, 3),
            "milliseconds": round(processing_time * 1000, 2)
        }
    
    return response

def create_error_response(error_message: str, error_code: str = "PROCESSING_ERROR") -> Dict:
    """
    Create a standardized error response.
    
    Args:
        error_message: Error description.
        error_code: Optional error code.
    
    Returns:
        Error response dict.
    """
    return {
        "success": False,
        "error": {
            "message": error_message,
            "code": error_code
        }
    }

def get_class_names() -> List[str]:
    """
    Get the list of brain tumor class names in model order.
    
    Returns:
        List of class names.
    """
    return ['no_tumor', 'glioma', 'meningioma', 'pituitary']

def get_class_info() -> Dict[str, Dict[str, str]]:
    """
    Get detailed information about each class.
    
    Returns:
        Dict with class details.
    """
    return {
        "no_tumor": {
            "name": "No Tumor",
            "description": "No tumor detected in the brain scan",
            "severity": "none"
        },
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
        "pituitary": {
            "name": "Pituitary Tumor",
            "description": "A tumor in the pituitary gland",
            "severity": "medium"
        }
    }