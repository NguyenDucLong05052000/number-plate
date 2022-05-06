from torch import nn
import torch
import cv2
import torch.utils.data as data
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3)),
            nn.ReLU(),
           
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=128,out_channels=125,kernel_size=(3,3)),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
                       
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(500, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 249),
            #nn.Softmax(1)
            
        )
    def forward(self, x):
        x = self.block(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class MyDataset(data.Dataset):
    def __init__(self,list_segment):  
        self.list_segment = list_segment
    def __len__(self):
        return len(self.list_segment)
    def __getitem__(self,index):
        image = self.list_segment[index]
        image = cv2.resize(image, (128, 128))
        image = torch.Tensor(image)
        image = torch.reshape(image,(3,128,128))
        return image
