
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

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



MODEL_PATH = "brain_tumor_cnn.pth"  # trained model
IMAGE_PATH = r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2026-01-06 143829.png"   #test image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def test_image(image_path, model_path):
    
    try:
        model = Brain_Resnet_18(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully")
        print(f"Device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
  
    print("\n[2/4] Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image loaded: {image_path}")
        print(f"Original size: {image.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print("\n[3/4] Preprocessing...")
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Preprocessed to: {image_tensor.shape}")
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return
    
    
    print("\n[4/4] Making prediction...")
    try:
        with torch.no_grad():
            output = model(image_tensor)
            probability = F.softmax(output, dim=1)
            
            # Get top prediction
            confidence, predicted = torch.max(probability, dim=1)
            
            # Get top 2
            top_2_prob, top_2_class = torch.topk(probability, k=2, dim=1)
        
        # Extract results
        primary_class = class_names[top_2_class[0][0].item()]
        primary_conf = top_2_prob[0][0].item() * 100
        
        secondary_class = class_names[top_2_class[0][1].item()]
        secondary_conf = top_2_prob[0][1].item() * 100
        
        
        print(f" Prediction complete!")
        
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\nPRIMARY PREDICTION:")
        print(f"   Class: {primary_class.upper()}")
        print(f"   Confidence: {primary_conf:.2f}%")
        
        if secondary_conf > 5:
            print(f"\SECONDARY PREDICTION:")
            print(f"   Class: {secondary_class.upper()}")
            print(f"   Confidence: {secondary_conf:.2f}%")
        else:
            print(f"\nModel is very confident in primary prediction!")
        
        print(f"\nALL CLASS PROBABILITIES:")
       
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_image(IMAGE_PATH, MODEL_PATH)
