import os
import PIL
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models
from src.model import Model

data_dir = '/home/haiwen/kaggle/data/histopathologic-cancer-detection'
model_dir = './experiments'
dataset = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize])

model = Model(base=models.resnet50)
model.half()
target = []
with torch.no_grad():
    checkpoint = torch.load(model_dir + "/best.pth")
    model.load_state_dict(checkpoint["model"])
    model.cuda()
    model.eval()
    for index, row in dataset.iterrows():
        name = row['id']

        path = os.path.join(data_dir, "test" , name+".tif")
        x = PIL.Image.open(path)
        x = test_transform(x)
        x = x.unsqueeze(0)

        pred = model(x.cuda()).sigmoid().cpu().numpy()
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        target.append(pred)
#target = []
#for sample_pred in result:
#    pred = []
#    for i,score in enumerate(sample_pred):
#        temp = 0
#        if score > 0.5:
#            temp = 1
#        pred.append(temp)

#    target.append(pred)
            #if score > cfg['thres'][i]:
            #    pred.append(str(i))
        #if len(pred) == 0:
            #pred.append(str(sample_pred.argmax()))
        #target.append(' '.join(pred))

#print(len(dataset))
#print(len(target[0]))
dataset['label'] = target
dataset.to_csv('submit.csv',index=False)