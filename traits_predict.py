import os

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset #, mean, std
from traits_config import TRAITS_KEYS, TRAITS_KEYS_MAP, DEVICE, models_weights
import torchvision.transforms as transforms

from models.mobilenet import MultiOutputModel_Mobilenet
from models.resnet import MultiOutputModel_Resnet
from models.squeezenet import MultiOutputModel_Squeezenet
from models.harmonicnet import MultiOutputModel_Harmonicnet
from models.efficientnet import MultiOutputModel_Efficientnet
from models.vitnet import MultiOutputModel_Vitnet

import cv2
import traits_utils as utils
from collections import Counter


models_types=[
    MultiOutputModel_Mobilenet,
    MultiOutputModel_Resnet,
    MultiOutputModel_Squeezenet,
    MultiOutputModel_Efficientnet,    
    MultiOutputModel_Harmonicnet,        
    MultiOutputModel_Vitnet
]


def checkpoint_load(model, name, verb=False):
    if verb:
        print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

def check_predict(real, predicted, weak=False):
    if not weak:
        return real==predicted
    return True if abs(real-predicted) < 2 else False  

def load_models(device, models_types):

    attributes = AttributesDataset(empty=True)
    models=[]
    for m in models_types:
        models.append(m(n_classes=attributes, pretrained=True).to(device))
    
    return models    
        
         
def predict(device, models, weights, image):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
          
    image = utils.resize_without_deforming_aspect_ratio(image)          
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension   
    
    predicted_matrix = [] 
    
    for m, w in zip(models, weights):
        checkpoint_load(m, w)
        

        with torch.no_grad():
            outputs = m(image) # get the predictions on the image
        
            predicted_vector = []
            
            for t in TRAITS_KEYS:               
                predicted_vector.append(outputs[t].cpu().max(1)[1].item())
                
            predicted_matrix.append(predicted_vector)                                
            
        
    predicted_matrix=utils.transpose(predicted_matrix)   
    #print(predicted_matrix)     
    
    result={}
    for t, p in zip(TRAITS_KEYS, predicted_matrix):
        c = Counter(p).most_common()
        #result[t]=(c, TRAITS_KEYS_MAP[t][1][int(c[0][0])], TRAITS_KEYS_MAP[t][0]) 
        result[t]=(TRAITS_KEYS_MAP[t][0], TRAITS_KEYS_MAP[t][1][int(c[0][0])], c[0][0]) 
        #print(f'{t}->{c}->{c[0][0]}->{TRAITS_KEYS_MAP[t][1][int(c[0][0])]}')
    
    return result        
        
    


if __name__ == '__main__':  
    
    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE == 'cuda' else "cpu") 
    
    models=load_models(device=device, models_types=models_types)
    
    #image_path = '/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits/ORLOV0/./after_5yo/stallion/Инструктор.jpg'
    image_path = '/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits/ORLOV0/./2yo/stallion/Зиновий слева.jpg'
    image = cv2.imread(image_path)   
    
    print(predict(device=device, models=models, weights=models_weights, image=image))
    
    