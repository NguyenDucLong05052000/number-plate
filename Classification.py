import cv2
from yolov3.configs import *
import torch
import torch.utils.data as data
from Network_OCR import NeuralNetwork


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

def Classification(list_segment):
    f=open('chars','r')
    arr=[]
    for i in f.readline():
        arr.append(i)
    test = MyDataset(list_segment)
    test_dataloader = data.DataLoader(test,batch_size=1,shuffle=False)
    model = NeuralNetwork()
    model.load_state_dict(torch.load('long.pth',map_location=torch.device('cpu')))
    s=''
    for i in test_dataloader:
        predict = arr[(model(i).argmax(1).item())]
        s=s+predict
        #print(predict)
    return s