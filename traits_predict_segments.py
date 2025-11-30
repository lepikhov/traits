import os
import sys

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset #, mean, std
import traits_config
import torchvision.transforms as transforms
import numpy as np

from dataset_segments import AttributesDataset

from models.mobilenet_segments import MultiOutputModel_Mobilenet
from models.resnet_segments import MultiOutputModel_Resnet
from models.squeezenet_segments import MultiOutputModel_Squeezenet
from models.harmonicnet_segments import MultiOutputModel_Harmonicnet
from models.efficientnet_segments import MultiOutputModel_Efficientnet
from models.vitnet_segments import MultiOutputModel_Vitnet

from PIL import Image
import cv2
import traits_utils as utils
from collections import Counter

import calculation

sys.path.append('../keypoints-for-entire-image')
import predict as segmentation

from traits_predict import checkpoint_load
from prepare_segments import merge_segments, clear_area


models_types=[
    #'mobilenet',
    #'resnet',
    #'squeezenet',
    #'efficientnet',   
    #'harmonicnet',     
    'vitnet', 
]

traits_segments_info=[
    (traits_config.TRAITS_HEAD_NECK_KEYS, ['Head', 'Neck'], None, traits_config.models_weights_Head_Neck),
    (traits_config.TRAITS_HEAD_NECK_BODY_KEYS, ['Head', 'Neck', 'Body'], None, traits_config.models_weights_Head_Neck_Body),
    (traits_config.TRAITS_REAR_LEG_KEYS, ['Rear leg'], None, traits_config.models_weights_Rear_leg),
    (traits_config.TRAITS_FRONT_LEG_KEYS, ['Front leg'], None, traits_config.models_weights_Front_leg), 
    (traits_config.TRAITS_BODY_KEYS, ['Body'], None, traits_config.models_weights_Body),
    (traits_config.TRAITS_BODY_FRONT_LEG_KEYS, ['Body', 'Front leg'], 'Rear leg', traits_config.models_weights_Body_Front_leg),
    (traits_config.TRAITS_BODY_NECK_KEYS, ['Body', 'Neck'], None, traits_config.models_weights_Body_Neck),
]

traits_segments_info_orlovskaya=[
    (traits_config.TRAITS_HEAD_NECK_KEYS, ['Head', 'Neck'], None, traits_config.models_weights_Head_Neck_orlovskaya),
    (traits_config.TRAITS_HEAD_NECK_BODY_KEYS, ['Head', 'Neck', 'Body'], None, traits_config.models_weights_Head_Neck_Body_orlovskaya),
    (traits_config.TRAITS_REAR_LEG_KEYS, ['Rear leg'], None, traits_config.models_weights_Rear_leg_orlovskaya),
    (traits_config.TRAITS_FRONT_LEG_KEYS, ['Front leg'], None, traits_config.models_weights_Front_leg_orlovskaya), 
    (traits_config.TRAITS_BODY_KEYS, ['Body'], None, traits_config.models_weights_Body_orlovskaya),
    (traits_config.TRAITS_BODY_FRONT_LEG_KEYS, ['Body', 'Front leg'], 'Rear leg', traits_config.models_weights_Body_Front_leg_orlovskaya),
    (traits_config.TRAITS_BODY_NECK_KEYS, ['Body', 'Neck'], None, traits_config.models_weights_Body_Neck_orlovskaya),
]

traits_type_info=(
    traits_config.TRAITS_TYPE_KEYS, ['Type'], traits_config.models_weights_Type
)  

     
def predict_segment(device, models, weights, traits_keys, attributes, segments, image):    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #convert PIL to cv2          
    cv2_image = np.array(image)
    # Convert RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()          
    cv2_image = utils.resize_without_deforming_aspect_ratio(cv2_image)          
    cv2_image = transform(cv2_image).to(device)
    cv2_image = cv2_image.unsqueeze(0) # add a batch dimension   
    
    predicted_matrix = [] 
    
    separator = " "
    segments_type = separator.join(segments)
    print(segments_type)
    
    for model_type, weights in zip(models, weights):
        match model_type:
            case 'mobilenet':
                model = MultiOutputModel_Mobilenet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)
            case 'squeezenet':
                model = MultiOutputModel_Squeezenet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)
            case 'resnet':    
                model = MultiOutputModel_Resnet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)
            case 'efficientnet':    
                model = MultiOutputModel_Efficientnet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)   
            case 'harmonicnet':    
                model = MultiOutputModel_Harmonicnet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)         
            case 'vitnet':    
                model = MultiOutputModel_Vitnet(n_classes=attributes, segments_type=segments_type, 
                                                traits_keys=traits_keys).to(device)                               
            case _:
                pass  

        checkpoint_load(model, weights)
              

        with torch.no_grad():
            outputs = model(cv2_image) # get the predictions on the image
        
            predicted_vector = []
            
            for t in traits_keys:               
                predicted_vector.append(outputs[t].cpu().max(1)[1].item())
                
            predicted_matrix.append(predicted_vector)                                
            
        
    predicted_matrix=utils.transpose(predicted_matrix)   
    print(predicted_matrix)     
    
    result={}
    for t, p in zip(traits_keys, predicted_matrix):
        
        #if t=='type': # for type use only vinet result
            #result[t]=(traits_config.TRAITS_KEYS_MAP[t][0], traits_config.TRAITS_KEYS_MAP[t][1][int(p[5])], p[5]) 
            #print(f'{t}->{p}->{p[5]}->{traits_config.TRAITS_KEYS_MAP[t][1][int(p[5])]}')
        #else:            
        c = Counter(p).most_common()
        #result[t]=(c, TRAITS_KEYS_MAP[t][1][int(c[0][0])], TRAITS_KEYS_MAP[t][0]) 
        result[t]=(traits_config.TRAITS_KEYS_MAP[t][0], traits_config.TRAITS_KEYS_MAP[t][1][int(c[0][0])], c[0][0])             
        print(f'{t}->{c}->{c[0][0]}->{traits_config.TRAITS_KEYS_MAP[t][1][int(c[0][0])]}')
    
    return result       

def predict_with_segments(device, image, breed=None): 
    
    segmentation_model, _ = segmentation.prepare_models()
    boxes, segments, _, _  = segmentation.get_segments(segmentation_model, image)
    
    traits={}
    
    match breed:
        case 'orlovskaya':
            info = traits_segments_info_orlovskaya
        case _:
            info = traits_segments_info

    
    for keys, segs, clear_seg, weights in info:
        bxs = merge_segments(segs, boxes, segments) 
        if not bxs==[]:    
            im=image.copy()        
            if clear_seg is not None:
                try:
                    clr_area = boxes[segments.index(clear_seg)]
                except:
                    clr_area = [0,0,0,0]                 
                im = clear_area(im, clr_area)            
            im = im.crop(bxs)     
            
            separator = " "
            attributes = AttributesDataset(segments_type=separator.join(segs))
            
            traits.update(predict_segment(device=device, models=models_types, weights=weights, traits_keys=keys, 
                            attributes=attributes, segments=segs, image=im))
              
    return traits

def calculate_traits(device, image, breed=None):     
    segmentation_model,  keypoints_models = segmentation.prepare_models()
    #kp, _, _, _, _ = segmentation.predict(segmentation_model, keypoints_models, image)
    kp, kp_image, _, _, _ = segmentation.predict(segmentation_model, keypoints_models, image, color_convert=True)
    #print(kp, len(kp))
    kps_image = utils.draw_keypoints_numbers(kp, kp_image)
    h, _, _ = kps_image.shape
    p1, p2 = calculation.get_horizont_line(kp)
    #print(p1, p2)
    pt1 = int(p1[0]), h-int(p1[1])
    pt2 = int(p2[0]), h-int(p2[1])
    kps_image = cv2.line(kps_image, pt1, pt2, color=(0,0,255), thickness=5)
    cv2.imwrite("./outputs/__with_keypoints.png", kps_image)
    return calculation.calculate_traits(kp, draw=True, image=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))    

def predict_type(device, image): 
    
    keys, segs, weights = traits_type_info
    
    separator = " "
    attributes = AttributesDataset(segments_type=separator.join(segs))
    
    return predict_segment(device=device, models=models_types, weights=weights, traits_keys=keys, 
                            attributes=attributes, segments=segs, image=image)

if __name__ == '__main__':  
    
    device = torch.device("cuda" if torch.cuda.is_available() and traits_config.DEVICE == 'cuda' else "cpu") 
    
    image_path = '/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits/ORLOV0/./after_5yo/stallion/Инструктор.jpg'
    #image_path = '/home/pavel/projects/horses/soft/python/morphometry/datasets/2024/traits/ORLOV0/./2yo/stallion/Зиновий слева.jpg'
    image = Image.open(image_path)   
    
    #print(predict_with_segments(device, image))
    #print(predict_type(device, image))
    
    calculated_traits = calculate_traits(device, image)
    
    print(calculated_traits)
    
    
    

    


            
            
    
    