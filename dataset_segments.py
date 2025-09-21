import os
import cv2

import warnings

import numpy as np
from torch.utils.data import Dataset

import data_loading
import torch
from data_loading import is_gray_scale
import traits_utils as utils
import traits_config
import pandas as pd


class AttributesDataset():
    def __init__(self, df=None, empty=False, segments_type=''):
            
            
        if not empty:
            match segments_type:
                case 'Type':
                    self.type_expressiveness = np.unique(df['type'])                         
                case 'Head Neck':
                    self.nape = np.unique(df['nape'])                    
                case 'Head Neck Body':                    
                    self.head_0 = np.unique(df['head_0'])
                    self.withers_0 = np.unique(df['withers_0'])
                    self.spine_0 = np.unique(df['spine_0']) 
                    self.withers_1 = np.unique(df['withers_1']) 
                    self.rib_cage_0 = np.unique(df['rib_cage_0'])  
                    self.angle_4 = np.unique(df['angle_4'])                                        
                case 'Rear leg': 
                    #self.hip = np.unique(df['hip'])
                    self.shin_0 = np.unique(df['shin_0']) 
                    self.tailstock = np.unique(df['tailstock'])   
                    self.angle_13 = np.unique(df['angle_13']) 
                    #self.angle_14 = np.unique(df['angle_14']) 
                    self.angle_15 = np.unique(df['angle_15'])                                      
                case 'Front leg':
                    self.headstock = np.unique(df['headstock'])
                    #self.angle_11 = np.unique(df['angle_11']) 
                    self.angle_12 = np.unique(df['angle_12'])                     
                case 'Body':        
                    self.rump = np.unique(df['rump'])    
                    self.spine_3 = np.unique(df['spine_3']) 
                    self.lower_back_0 = np.unique(df['lower_back_0']) 
                    self.angle_10 = np.unique(df['angle_10'])            
                case 'Body Front leg': 
                    self.shoulder = np.unique(df['shoulder']) 
                    self.falserib_0 = np.unique(df['falserib_0'])
                    self.forearm = np.unique(df['forearm'])   
                    self.angle_5 = np.unique(df['angle_5'])                  
                case 'Body Neck':  
                    self.neck_0 = np.unique(df['neck_0'])  
                    self.angle_4 = np.unique(df['angle_4'])                   
                case _:
                    pass

        match segments_type:
            case 'Type':
                self.num_type_expressiveness = 6#len(self.type_expressiveness)              
            case 'Head Neck':
                self.num_nape = 4#len(self.nape)                  
            case 'Head Neck Body':                    
                self.num_head_0 = 4#len(self.head_0)
                self.num_withers_0 = 4#len(self.withers_0)  
                self.num_spine_0 = 4#len(self.spine_0)      
                self.num_withers_1 = 4#len(self.withers_1)   
                self.num_rib_cage_0 = 4#len(self.rib_cage_0)    
                self.num_angle_4 = 4#len(self.angle_4)                                       
            case 'Rear leg': 
                self.num_hip = 4#len(self.headstock)
                self.num_shin_0 = 4#len(self.shin_0)  
                self.num_tailstock = 4#len(self.tailstock)     
                self.num_angle_13 = 4#len(self.angle_13)
                self.num_angle_14 = 4#len(self.angle_14)
                self.num_angle_15 = 4#len(self.angle_15)                                                 
            case 'Front leg':
                self.num_headstock = 4#len(self.headstock)   
                self.num_angle_11 = 4#len(self.angle_11)
                self.num_angle_12 = 4#len(self.angle_12)                                                    
            case 'Body':        
                self.num_rump = 4#len(self.rump)
                self.num_spine_3 = 4#len(self.spine_3)
                self.num_lower_back_0 = 4#len(self.lower_back_0)  
                self.num_angle_10 = 4#len(self.angle_10)              
            case 'Body Front leg': 
                self.num_shoulder = 4#len(self.shoulder)
                self.num_falserib_0 = 4#len(self.falserib_0)
                self.num_forearm = 4#len(self.forearm)  
                self.num_angle_5 = 4#len(self.angle_5)              
            case 'Body Neck':  
                self.num_neck_0 = 4#len(self.neck_0) 
                self.num_angle_4 = 4#len(self.angle_4)                                   
            case _:
                pass            
                                                                        

class TraitsDataset(Dataset):
    def __init__(self, df, attributes, traits_keys, transform=None):
        super().__init__()

        self.transform = transform
        self.df = df      
        self.attributes = attributes 
        self.traits_keys = traits_keys
        

    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx, transformed_image_to_file=False):
        # take the data sample by its index
        filepath = self.df.iloc[idx]['imagedir']        
        filename = self.df.iloc[idx]['imagefile']       
        img_name = os.path.join(filepath, filename)
                        
        image = cv2.imread(img_name)                
        # If the image is Greyscale convert it to RGB
        #gr, _ = is_gray_scale(image)
        #if gr:
        #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = utils.resize_without_deforming_aspect_ratio(image) # resize without deforming aspect ratio
        
        
        
        
        
        if transformed_image_to_file:
            cv2.imwrite(f'./outputs/{filename}', image)  
            

        # apply the image augmentations if needed
                    
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']    
                            

        labels={}
        for t in self.traits_keys:
            label = self.df.iloc[idx][t]
            label = torch.as_tensor(label, dtype=torch.long)
            labels[t] = label
            
        # return the image and all the associated labels
        dict_data = {
            'image': image,
            'labels': labels
        }
        return dict_data            
    
if __name__ == "__main__":
    df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_head_neck.json'), orient='table')    
    attributes = AttributesDataset(df, segments='Head Neck')    
    train_dataset = TraitsDataset(df, attributes=attributes, traits_keys=traits_config.TRAITS_HEAD_NECK_KEYS)


    print(len(train_dataset))    
    for i in range(10): #len(train_dataset)):
        print(i, train_dataset.__getitem__(i, transformed_image_to_file=True)['labels'])
    print(train_dataset.__len__())
    print(attributes.nape)
