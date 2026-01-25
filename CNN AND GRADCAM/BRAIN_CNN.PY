import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
# import optuna as optuna
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision


# preparing the data 
train_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])

])
test_transforms = transforms.Compose([
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# now using the dataloader and dataset class
train_dataset = ImageFolder(r'C:\Users\Lenovo\OneDrive\Desktop\brain tumor dataset\Training',transform=train_transforms)
test_dataset = ImageFolder(r'C:\Users\Lenovo\OneDrive\Desktop\brain tumor dataset\Testing',transform=test_transforms)

train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=2)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False,num_workers=0)

# now defining the basic architecture of our model which will use resnet method

class Resnet_basic(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride = 1):
        super(Resnet_basic,self).__init__() 
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.Relu1 = nn.ReLU(inplace=True)

        # second convolutional layer
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch2 = nn.BatchNorm2d(out_channels)

        # now adding shortcut line
        self.shortcut = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.Relu1(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out += self.shortcut(x)
        out = self.Relu1(out)
        return out 
    

# now we will define the actual class of out model 

class Brain_Resnet_18(nn.Module):
    def __init__(self,num_classes = 4):
        super(Brain_Resnet_18,self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.lay1 = self._make_layer(Resnet_basic,64,2,stride =1)
        self.lay2 = self._make_layer(Resnet_basic,128,2,stride =2)
        self.lay3 = self._make_layer(Resnet_basic,256,2,stride =2)
        self.lay4 = self._make_layer(Resnet_basic,512,2,stride =2)

        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(0.4)
        self.fully_connected = nn.Linear(512,num_classes)
        
    
    # make_layer function
    
    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,out_channels,stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    

    def forward(self,x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        out = self.lay1(out)
        out = self.lay2(out)
        out = self.lay3(out)
        out = self.lay4(out)

        out = self.average_pool(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.fully_connected(out)
        return out


if __name__ == '__main__':
    epoch = 5
    learning_rate = 0.01

    model = Brain_Resnet_18()
    model = model.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # now defining cross entropy loss
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(),lr = learning_rate,weight_decay=1e-4)

    # training loop
    try:
        for epochs in range(epoch):
            model.train()
            running_loss = 0
            total = 0
            correct = 0

            for image,label in train_loader:
                image,label = image.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')),label.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

                # zero gradient 
                optimiser.zero_grad()

                # forward pass
                outputs = model(image)
                loss = criterion(outputs,label)

                # back propagation
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                # now we will output top 2 predicted 
                probability = torch.softmax(outputs,dim=1)
                _, predicted = torch.max(probability,1)
                total+=label.size(0)
                correct = correct + (predicted==label).sum().item()

                epoch_loss = running_loss/len(train_loader)
                epoch_acc = 100 *(correct/total)
                print(f'epoch - {epochs+1} loss - {epoch_loss} accuracy - {epoch_acc}')
                torch.save(model.state_dict(),"brain_tumor_cnn.pth")

    # to stop training manually
    except KeyboardInterrupt:
        print("Training stopped manually")
        print("Model is saved")
        torch.save(model.state_dict(),"brain_tumor_cnn.pth")


    model.eval()
    test_correct = 0
    test_total = 0

    # now loading model to test 

    with torch.no_grad():

        for image,label in test_loader:
            image,label = image.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')),label.to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(image)
            probability = torch.softmax(outputs.data,1)
            confidence,predicted = torch.max(probability,1) 
            top_2,top_class = torch.topk(probability,k=2,dim=1)
            test_total +=label.size(0)
            test_correct = test_correct + (predicted==label).sum().item()

        test_accuracy = 100*(test_correct/test_total)
        print(f'test accuracy = {test_accuracy}')


    # now checking our model on different images
    def test_new_img(image_path,model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((96,96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
           output = model(image_tensor)
           probability = F.softmax(output,1)

           confidence,predicted = torch.max(probability,dim=1)

           top_2,top_class = torch.topk(probability,k=2,dim=1)
        
        class_names = train_dataset.classes

        # now defining top 2 predictions
        primary_class = class_names[top_class[0][0].item()]
        primary_conf = top_2[0][0].item()*100

        secondary_class = class_names[top_class[0][1].item()]
        secondary_conf = top_2[0][1].item()*100

        print(f"primary class - {primary_class} ({primary_conf:.2f}%)")

        if secondary_conf>5:
            print(f"secondary class - {secondary_class} ({secondary_conf:.2f}%)")
        else:
            print("Model is confident on primary output")
        
        return {
            "primary" : (primary_class,primary_conf),
            "secondary" : (secondary_class,secondary_conf)
        }
    
    test_new_img(r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2025-12-24 125604.png",model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    




        
