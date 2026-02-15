import torch
import numpy as np
import cv2
from typing import Tuple, Optional
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def apply_gradcam_library(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_layer: torch.nn.Module,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    cam = GradCAM(
        model=model,
        target_layers=[target_layer],
    )
    
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    
    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )
    
    grayscale_cam = grayscale_cam[0]
    
    if original_image.max() > 1.0:
        rgb_img = original_image.astype(np.float32) / 255.0
    else:
        rgb_img = original_image.astype(np.float32)
    
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (rgb_img.shape[1], rgb_img.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5
    )
    
    heatmap_uint8 = np.uint8(255 * grayscale_cam)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_colored, visualization