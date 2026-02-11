import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import numpy as np
import cv2 as cv
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import torch.nn.functional as F
import optuna as optuna
# now we will define some image pre processing methods 

train_transformers = transforms.Compose([
    transforms.RandomResizedCrop(96,scale=(0.8,1.0)), # we did resizing of images x will be 224 and y will be of 224 pixels
    transforms.RandomHorizontalFlip(), # randomly flipping images to have more variation
    transforms.ColorJitter(brightness=0.2,contrast=0.2),
    transforms.RandomRotation(10), #randomly rotating image by 10 degrees
    transforms.ToTensor(), # now we have to convert image to tensor 
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]) # 3 values because of 3 dimensions of image (length breadth and color channels)

])

test_transformers = transforms.Compose([
    transforms.RandomResizedCrop(96,scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std = [0.229,0.224,0.225])
])
# now loading dataset with imagefolder techniques

train_dataset = ImageFolder(root=r"C:\Users\Lenovo\OneDrive\文档\train",transform=train_transformers)
test_dataset = ImageFolder(root=r"C:\Users\Lenovo\OneDrive\文档\test",transform=test_transformers)

# now using dataloader class we will officially load the dataset to this project

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=2) # we used Dataloader class ,used batch size of 32 with shuffle false and numworkers is 2 for faster training
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=0) # same for test loader

# now we will define the nn architecture

class ConvolutionalNN(nn.Module):
    
    def __init__(self):
        super(ConvolutionalNN,self).__init__()

    # now making architecture
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1) # our first convolutional layer
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)# second convolutional layer
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=80,kernel_size=3,padding=1) # 3rd convolutional layer
        self.batchnorm3 = nn.BatchNorm2d(80)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=80,out_channels=160,kernel_size=3,padding=1) # 4th convolutional layer 
        self.batchnorm4 = nn.BatchNorm2d(160)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(in_channels=160,out_channels=220,kernel_size=3,padding=1)
        self.batchnorm5 = nn.BatchNorm2d(220)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2,2)

        # now making fully connected layer 
        self.pool_global = nn.AdaptiveAvgPool2d((1,1))
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(220,2)

    

    # now forward pass
    def forward(self,x):
        # now we will apply forward pass for layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # forward pass for layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # for layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # layer 4
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # layer 5
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = self.pool_global(x)
        x = torch.flatten(x,1)
        x = self.dropout1(x)
        x = self.fc(x)
        

        return x

if __name__ == '__main__':
    epoch = 35
    learning_rate = 0.001

    model = ConvolutionalNN()
    model = model.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# now defininng loss function
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(),lr = learning_rate,weight_decay=1e-4)

# now training loop 
    try:
        for epochs in range(epoch):
            model.train()
            running_loss = 0
            total = 0
            correct = 0

            for image,labels in train_loader:
        # moving data to device
                image,labels = image.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),labels.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # zero the gradients to avoid accumulation of gradients
                optimiser.zero_grad()

        # forward pass
                outputs = model(image)
                loss = criterion(outputs,labels)

        #backward pass
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data,1)
                total+=labels.size(0)
                correct = correct + (predicted==labels).sum().item()
                
    

            epoch_loss = running_loss/len(train_loader)
            epoch_acc = 100*correct/total # percentage correct
            print(f'epoch {epochs+1} , loss {epoch_loss},accuracy {epoch_acc}')
            torch.save(model.state_dict(),"cat_vs_dog.pth")

    except KeyboardInterrupt:
        print("Training stopped manually . Model saved")
        torch.save(model.state_dict(),"cat_vs_dog.pth")




    model.eval()
    test_correct = 0
    test_total = 0

# now testing it on test sample
    with torch.no_grad():
        for image,labels in test_loader:
            image,labels = image.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),labels.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(image)
            test_loss = criterion(outputs,labels)

            _, predicted = torch.max(outputs.data,1)
            test_total+=labels.size(0)
            test_correct += (predicted==labels).sum().item()

    
        test_accuracy = 100*test_correct/test_total
        print(f'test accuracy {test_accuracy}')
        

    

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
            probability = F.softmax(output,dim=1)
            confidence , predicted = torch.max(probability,1)

        class_names = train_dataset.classes
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()*100

        print(f'Prediction : {predicted_class}')
        print(f'confidence {confidence_score}')

        return predicted_class ,confidence_score

    test_new_img(r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2025-12-19 123016.png",model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    test_new_img(r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2025-12-19 170015.png",model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    test_new_img(r"C:\Users\Lenovo\OneDrive\图片\Screenshots\Screenshot 2025-12-19 170053.png",model,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    torch.save(model.state_dict(),"cat_vs_dog.pth")
