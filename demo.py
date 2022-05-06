from Detectnumberplate import detect_number_plate
from Segmentation_characterJP import segment_char
from Classification import Classification


# def blur_all_image(img):
#     mask_size = np.random.choice([1])
#     blur = cv2.GaussianBlur(img,(mask_size, mask_size),3)
#     #blur=cv2.bilateralFilter(img, 9, 75, 75)
#     return blur

# def segment_char(imageori,image_path):
#     arr = []
#     file_name = os.path.basename(image_path)
#     imageori = cv2.resize(imageori,(224,128))

#     image_top = imageori[0:(int)(0.46*imageori.shape[0])]
#     # image_top1 = cv2.resize(image_top,(224,128))
#     image_topGray =cv2.cvtColor(image_top,cv2.COLOR_BGR2GRAY)
#     # image_topGray = cv2.resize(image_topGray,(224,128))
#     image_topGray = blur_all_image(image_topGray)
#     _,image_topGray = cv2.threshold(image_topGray,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,1)
#     contours, hierarchy = cv2.findContours(image_topGray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours,key= lambda x:cv2.boundingRect(x)[0])
#     temp = 0
#     for c in contours:
#         x,y,w,h = cv2.boundingRect(c)
#         if(h>0.4*image_topGray.shape[0] and w<0.5*image_topGray.shape[1]):
#             temp=temp+1
#             crop_img = image_top[y:y+h, x:x+w]
#             arr.append(crop_img)
#             #cv2.imwrite(f"./IMAGES_SEGMENT/cropping_image_{file_name}_{temp}",crop_img)
    
#     image_bot = imageori[(int)(0.35*imageori.shape[0]):imageori.shape[0]]
#     # image_bot1 = cv2.resize(image_bot,(224,128))
#     image_botGray =cv2.cvtColor(image_bot,cv2.COLOR_BGR2GRAY)
#     # image_botGray = cv2.resize(image_botGray,(224,128))
#     image_botGray = blur_all_image(image_botGray)
#     _,image_botGray = cv2.threshold(image_botGray,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,1)
#     contours, hierarchy = cv2.findContours(image_botGray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#     contours = sorted(contours,key= lambda x:cv2.boundingRect(x)[0])
#     for c in contours:
#         x,y,w,h = cv2.boundingRect(c)
#         if((h>0.5*image_botGray.shape[0] and w<0.5*image_botGray.shape[1] )
#         or(w*h>20 and x>0.5*image_botGray.shape[1] and y>0.3*image_botGray.shape[0] and x+w<0.67*image_botGray.shape[1] and y+h<image_botGray.shape[0]*0.68 )
#         or (w*h>20 and x>0.02*image_botGray.shape[1] and y>0.2*image_botGray.shape[0]) and x+w<0.3*image_botGray.shape[1] and y+h<image_botGray.shape[0]*0.8 ):    
#     #         #image_bot = cv2.rectangle(image_bot,(x,y),(x+w,y+h),(125,125,125),2)
#     #         print('x,y,w,h =',x,' ',y,' ',w,' ',h)
#             temp=temp+1
#             crop_img = image_bot[y:y+h, x:x+w]
#             arr.append(crop_img)
#             #print(crop_img.shape)
#             #cv2.imwrite(f"./IMAGES_SEGMENT/cropping_image_{file_name}_{temp}",crop_img)
#     return arr

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork,self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3)),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3)),
#             nn.ReLU(),
            
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3)),
#             nn.ReLU(),
           
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(in_channels=128,out_channels=125,kernel_size=(3,3)),
#             nn.ReLU(),
            
#             nn.MaxPool2d(2),
                       
#         )
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(500, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 249),
#             #nn.Softmax(1)
            
#         )
#     def forward(self, x):
#         x = self.block(x)
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# class MyDataset(data.Dataset):
#     def __init__(self,list_segment):  
#         self.list_segment = list_segment
#     def __len__(self):
#         return len(self.list_segment)
#     def __getitem__(self,index):
#         image = self.list_segment[index]
#         image = cv2.resize(image, (128, 128))
#         image = torch.Tensor(image)
#         image = torch.reshape(image,(3,128,128))
#         return image

# def Classification(list_segment):
#     f=open('chars','r')
#     arr=[]
#     for i in f.readline():
#         arr.append(i)
#     test = MyDataset(list_segment)
#     test_dataloader = data.DataLoader(test,batch_size=1,shuffle=False)
#     model = NeuralNetwork()
#     model.load_state_dict(torch.load('long.pth',map_location=torch.device('cpu')))
#     s=''
#     for i in test_dataloader:
#         predict = arr[(model(i).argmax(1).item())]
#         s=s+predict
#         #print(predict)
#     return s


def Full_module(image_path):
    img = detect_number_plate(image_path)
    list_segment= segment_char(img,image_path)
    result = Classification(list_segment)
    return result



