import os
import cv2

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config import TRAITS_KEYS, ROOT_DATA_DIRECTORY 

import data_loading
import torch


class AttributesDataset():
    def __init__(self, df):
            self.head_0 = np.unique(df['head_0'])
            self.nape = np.unique(df['nape'])
            self.neck_0 = np.unique(df['neck_0']) 
            self.withers_0 = np.unique(df['withers_0']) 
            self.shoulder = np.unique(df['shoulder']) 
            self.spine_0 = np.unique(df['spine_0']) 
            self.rump = np.unique(df['rump']) 
            self.falserib_0 = np.unique(df['falserib_0'])
            self.forearm = np.unique(df['forearm']) 
            self.headstock = np.unique(df['headstock'])
            self.hip = np.unique(df['headstock']) 
            self.shin_0 = np.unique(df['shin_0']) 
            self.tailstock = np.unique(df['tailstock']) 
            self.withers_1 = np.unique(df['withers_1']) 
            self.spine_3 = np.unique(df['spine_3']) 
            self.lower_back_0 = np.unique(df['lower_back_0']) 
            self.rib_cage_0 = np.unique(df['rib_cage_0']) 
            self.angle_4 = np.unique(df['angle_4']) 
            self.angle_5 = np.unique(df['angle_5']) 
            self.angle_10 = np.unique(df['angle_10']) 
            self.angle_11 = np.unique(df['angle_11']) 
            self.angle_12 = np.unique(df['angle_12']) 
            self.angle_13 = np.unique(df['angle_13']) 
            self.angle_14 = np.unique(df['angle_14']) 
            self.angle_15 = np.unique(df['angle_15'])

            self.num_head_0 = 4#len(self.head_0)
            self.num_nape = 4#len(self.nape)
            self.num_neck_0 = 4#len(self.neck_0)
            self.num_withers_0 = 4#len(self.withers_0)
            self.num_shoulder = 4#len(self.shoulder)
            self.num_spine_0 = 4#len(self.spine_0)
            self.num_rump = 4#len(self.rump)
            self.num_falserib_0 = 4#len(self.falserib_0)
            self.num_forearm = 4#len(self.forearm)
            self.num_headstock = 4#len(self.headstock)
            self.num_hip = 4#len(self.headstock)
            self.num_shin_0 = 4#len(self.shin_0)
            self.num_tailstock = 4#len(self.tailstock)
            self.num_withers_1 = 4#len(self.withers_1)
            self.num_spine_3 = 4#len(self.spine_3)
            self.num_lower_back_0 = 4#len(self.lower_back_0)
            self.num_rib_cage_0 = 4#len(self.rib_cage_0)
            self.num_angle_4 = 4#len(self.angle_4)
            self.num_angle_5 = 4#len(self.angle_5)
            self.num_angle_10 = 4#len(self.angle_10)
            self.num_angle_11 = 4#len(self.angle_11)
            self.num_angle_12 = 4#len(self.angle_12)
            self.num_angle_13 = 4#len(self.angle_13)
            self.num_angle_14 = 4#len(self.angle_14)
            self.num_angle_15 = 4#len(self.angle_15)


class TraitsDataset(Dataset):
    def __init__(self, df, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.df = df      
        self.attributes = attributes 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # take the data sample by its index
        filepath = self.df.iloc[idx]['imagedir']        
        filename = self.df.iloc[idx]['imagefile']       
        img_name = os.path.join(filepath, filename)

        print(filepath, filename)
        image = Image.open(img_name) #cv2.imread(img_name)
        print(image.size)
        # If the image is Greyscale convert it to RGB
        #gr, _ = data_loading.is_gray_scale(image)
        #if gr:
        #    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # apply the image augmentations if needed
        if self.transform:
            image = self.transform(image)
            print(image.size())

        labels={}
        for i in range(len(TRAITS_KEYS)):
            label = self.df.iloc[idx][TRAITS_KEYS[i]]
            label = torch.as_tensor(label, dtype=torch.long)
            labels[TRAITS_KEYS[i]] = label
            

        # return the image and all the associated labels
        dict_data = {
            'image': image,
            'labels': labels
        }
        return dict_data            
    
if __name__ == "__main__":
    df = data_loading.tps_list()     
    attributes = AttributesDataset(df)    
    train_dataset = TraitsDataset(df, attributes=attributes)


    print(len(train_dataset))    
    for i in range(len(train_dataset)):
        print(i, train_dataset.__getitem__(i))
    #print(train_dataset.__len__())
    #print(attributes.angle_4)
