import os
from gradcam_lib import apply_gradcam_library
from model import load_model, get_model
from utils_pytorch import preprocess_for_model, load_image_from_bytes
from PIL import Image

# Load everything
load_model("brain_tumor_cnn.pth")
model_wrapper = get_model()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

img_path = os.path.join(BASE_DIR, r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2026-01-06 143829.png")
# Load test image
with open(img_path, "rb") as f:
    image_bytes = f.read()

pil_image, original_array = load_image_from_bytes(image_bytes)
image_tensor = preprocess_for_model(pil_image)

# Generate GradCAM
heatmap, overlay = apply_gradcam_library(
    model=model_wrapper.get_model(),
    image_tensor=image_tensor,
    original_image=original_array,
    target_layer=model_wrapper.get_target_layer()
)

# Save resultr
Image.fromarray(overlay).save("gradcam_result.png")
print("GradCAM saved!")