from __future__ import division, print_function

import json
import os
import uuid
# Ignore warnings
import warnings


import albumentations.augmentations.functional as F
import config
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image




warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def statistics(df, trait):
    categories = []
    categories_counters = {}
    
    print('')
    print('statistic for ', trait, ':')
    
    items = df[trait]
    
    for item in items:
        if not item in categories:
            categories.append(item)
            categories_counters.update({item:1})
        else:
            categories_counters[item] += 1
            
    categories.sort()
    sorted_categories_counters = {i: categories_counters[i] for i in categories}          
                        
    print(sorted_categories_counters)


def check_traits(traits):
    cnt = 0

    for key in traits:
        cnt += 1
        if key in config.TRAITS_KEYS:
            continue 
        if key in config.TRAITS_KEYS_AUX:        
            continue
        if key in config.TRAITS_KEYS_SERVICE:        
            continue
        return False
    
    return (False, True)[cnt >= len(config.TRAITS_KEYS)]      
    
# help-function for search patterns like 'IMAGE', 'LM', etc
# and its indexes in tps-file 
def search(pattern, text):
  res = []
  indxs = []
  i = 0
  for line in text:
    if pattern in line:
       res.append(line)
       indxs.append(i)
    i +=1   

  return res, indxs

#help function for replace point->comma and convert to float 
def comma_float(s):
  try:
    return float(s.replace(',','.'))
  except:
    return np.NaN 

def tps_list():

    os.system('./tree_script.sh')

    tps_files = []
    with open('filelist.txt') as fp:
        pathes = fp.readlines()
        for path in pathes:
            dir, filename = os.path.split(path)
            d = {'dir': os.path.join(config.ROOT_DATA_DIRECTORY,dir[2::]), 
                    'file' : filename[:len(filename)-1:]
                }
            tps_files.append(d)          

    for i, file in enumerate(tps_files):
        path = os.path.join(file['dir'],file['file'])
        print(path) 

    os.system('rm filelist.txt')          

    #create empty dataframe
    df=pd.DataFrame(columns=['id','landmarks','imagedir','imagefile'])  
    
    wrong_traits_list_count = 0  
    correct_traits_list_count = 0  

    for file in tps_files:

        dir = file['dir']
        file_name = os.path.join(dir,file['file'])

        with open(file_name, encoding="cp1251") as file:
            lines=file.readlines()
        images, _ = search('IMAGE',lines)
        #ids, _ = search('ID',lines)
        lm, lmixs = search('LM',lines)
        traits_strings, _ = search('COMMENT',lines)


        i=0
        for inx in lmixs:
            try:
                num = int(lines[inx][3:])
            except:
                print(lines[inx], inx)    
            if (inx+num+1) < len(lines):
                pnts = lines[inx+1:inx+1+num]
                #print(pnts)
            ps=[]
            for p in pnts:
                pnt =  list(map(comma_float, p.split(sep=' ')))
                ps.append(pnt) 

            try:
                relpath = images[i][6:-1]
            except:
                print(i)    
            relpath = relpath.replace('\\', '/')
            imagefile = os.path.basename(relpath)
            #path = os.path.join(dir,relpath[1:len(relpath)-len(imagefile)-1])
            path = os.path.join(dir,relpath[0:len(relpath)-len(imagefile)])
            #print(path)


            #if num != 72:
            #    print(file_name,'| img:', imagefile, '| num points:', num)    

            try:
                traits = json.loads(traits_strings[i][8:-1].replace('\'','"'))
            except:
                pass
                #print('bad traits strings for:', imagefile, ':' , traits_strings)
                                
            if not check_traits(traits):
               print(file_name,'| img:', imagefile, '| wrong traits list')  
               print(traits)
               wrong_traits_list_count += 1 
            else:
                correct_traits_list_count += 1                    

                df_traits = pd.DataFrame(data={k : v for k,v in filter(lambda t: t[0] in config.TRAITS_KEYS, traits.items())}, index=[0])        
                df_traits = df_traits.applymap(lambda x: 0 if x == -9 else x)                

                df_base = pd.DataFrame(
                {'id': uuid.uuid4().hex, 
                    'imagedir':path, 
                    'imagefile': imagefile,
                    'landmarks': [None],                             
                }, index=[0]
                )
                df_base.loc[0,'landmarks'] = ps

                df = df.append(pd.concat([df_base, df_traits], axis=1),
                                ignore_index=True)    
            
                i += 1
        print(file_name, " items:", i)            

    print('total number:\n', df.count())
    print('number of landmarks:\n', len(df['landmarks'][0]))
    print('number of correct traits list:', correct_traits_list_count)
    print('number of wrong traits list:', wrong_traits_list_count)

    return df


#Utility for grayscale detection
def is_gray_scale(img):
    if len(img.shape) < 3: return True, 1
    if img.shape[2]  == 1: return True, 2
    #b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    #if (b==g).all() and (b==r).all(): return True, 3
    return False, 0


# Utility for reading an image and for getting its annotations.
def get_horse(sdf, id):
    path=os.path.join(sdf.iloc[id]['imagedir'],sdf.iloc[id]['imagefile'])
    _, ext = os.path.splitext(path)
    if ext in ['.gif', '.GIF']:
        gif = imageio.mimread(path)            
        # convert form RGB to BGR 
        img_data = cv2.cvtColor(gif[0], cv2.COLOR_RGB2BGR)
    else:
        img_data = cv2.imread(path)
    #If the image is Greyscale convert it to RGB
    gr, _ = is_gray_scale(img_data)
    if gr:       
        img_data=cv2.cvtColor(img_data,cv2.COLOR_GRAY2RGB)
    # If the image is RGBA convert it to RGB.
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert("RGB"))        
    return img_data


       

if __name__ == "__main__":
    tps_df=tps_list()    
    print(tps_df.head(31))
    print(tps_df.columns)
    
    for trait in config.TRAITS_KEYS:
        statistics(tps_df, trait)
