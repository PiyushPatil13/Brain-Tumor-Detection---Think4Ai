import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# Model Architecture (from your original code)
class Resnet_basic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Resnet_basic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.Relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.Relu1(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out += self.shortcut(x)
        out = self.Relu1(out)
        return out

class Brain_Resnet_18(nn.Module):
    def __init__(self, num_classes=4):
        super(Brain_Resnet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.lay1 = self._make_layer(Resnet_basic, 64, 2, stride=1)
        self.lay2 = self._make_layer(Resnet_basic, 128, 2, stride=2)
        self.lay3 = self._make_layer(Resnet_basic, 256, 2, stride=2)
        self.lay4 = self._make_layer(Resnet_basic, 512, 2, stride=2)
        
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)
        self.fully_connected = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)
        
        out = self.average_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fully_connected(out)
        return out

# Model Loader Class
class BrainTumorModel:
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.model = Brain_Resnet_18(num_classes=4)
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
        
        print("Model loaded successfully!")
    
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = F.softmax(output, dim=1)
            
            confidence, predicted = torch.max(probability, dim=1)
            
            top_2_prob, top_2_class = torch.topk(probability, k=2, dim=1)
        
        primary_class = self.class_names[top_2_class[0][0].item()]
        primary_conf = top_2_prob[0][0].item() * 100
        
        secondary_class = self.class_names[top_2_class[0][1].item()]
        secondary_conf = top_2_prob[0][1].item() * 100
        
        all_predictions = []
        for idx, prob in enumerate(probability[0]):
            all_predictions.append({
                "class": self.class_names[idx],
                "probability": prob.item(),
                "percentage": f"{prob.item() * 100:.2f}%"
            })
        
        return {
            "predicted_class": primary_class,
            "confidence": primary_conf,
            "secondary_class": secondary_class,
            "secondary_confidence": secondary_conf,
            "all_predictions": all_predictions,
            "raw_output": output.cpu().numpy().tolist()
        }
    
    def get_target_layer(self):
        return self.model.lay4
    
    def get_model(self):
        return self.model

# Global instance
_model_instance = None

def load_model(model_path: str = "best_brain_tumor_cnn.pth"):
    global _model_instance
    if _model_instance is None:
        _model_instance = BrainTumorModel(model_path)
    return _model_instance

def get_model() -> BrainTumorModel:
    global _model_instance
    if _model_instance is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model_instance