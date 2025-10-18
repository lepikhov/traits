from __future__ import division, print_function

import json
import os
import uuid
# Ignore warnings
import warnings


import albumentations.augmentations.functional as F
import traits_config
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import math




warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def statistics(df, trait):
    categories = []
    categories_counters = {}
    
    print('')
    print('statistic for ', trait, ':')
    
    if trait in df:

        items = df[trait]
    
        for item in items:   
            if not item in categories:
                categories.append(item)
                categories_counters.update({item:1})
            else:
                categories_counters[item] += 1
            
    sum = 0
    for item in categories_counters.values():
        sum = sum + item            
     
    categories.sort()      

        
    sorted_categories_counters = {i: categories_counters[i] for i in categories}     
    sorted_categories_counters_percents =  {i: 100*categories_counters[i]/sum for i in categories}     
                    
    
    return sorted_categories_counters, sorted_categories_counters_percents 


def check_traits(traits):
    cnt = 0

    for key in traits:
        cnt += 1
        if key in traits_config.TRAITS_KEYS:
            continue 
        if key in traits_config.TRAITS_KEYS_AUX:        
            continue
        if key in traits_config.TRAITS_KEYS_SERVICE:        
            continue
        print('BAD TRAIT NAME: ', key)            
        return False
    
    return (False, True)[cnt >= len(traits_config.TRAITS_KEYS)]     

def check_traits_param(traits, traits_keys = traits_config.TRAITS_KEYS):
    cnt = 0

    for key in traits:
        cnt += 1
        if key in traits_keys:
            continue 
        print('BAD TRAIT NAME: ', key)            
        return False
    
    return True   
    
    
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

def tps_list(traits_keys = traits_config.TRAITS_KEYS, 
             traits_keys_excluded = [], 
             with_types = False,
             root_data_directory = traits_config.ROOT_DATA_DIRECTORY):

    command = './tree_script.sh "'+root_data_directory+'"'
    os.system(command)

    tps_files = []
    with open('filelist.txt') as fp:
        pathes = fp.readlines()
        for path in pathes:
            dir, filename = os.path.split(path)
            d = {'dir': os.path.join(root_data_directory,dir[2::]), 
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

        try:
            with open(file_name, encoding=traits_config.TPS_ENCODING) as file:        
                lines=file.readlines()
        except:
            with open(file_name, encoding=traits_config.ALTERNATIVE_TPS_ENCODING) as file:        
                lines=file.readlines()     
                                       
        images, _ = search('IMAGE',lines)
        #ids, _ = search('ID',lines)
        _, lmixs = search('LM',lines)
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
                                
            if not check_traits_param(traits, traits_keys = traits_keys + traits_keys_excluded):
               print(file_name,'| img:', imagefile, '| wrong traits list')  
               print(traits)
               wrong_traits_list_count += 1 
               break
            else:
                
                if with_types: 
                    if not 'type' in traits: 
                        continue
                 
                correct_traits_list_count += 1       
                                                  
                df_traits = pd.DataFrame(data={k : v for k,v in filter(lambda t: t[0] in traits_keys, traits.items())}, index=[0])                        
                df_traits = df_traits.applymap(lambda x: 0 if x == -9 else x)                

                df_base = pd.DataFrame(
                {'id': uuid.uuid4().hex, 
                    'imagedir':path, 
                    'imagefile': imagefile,
                    'landmarks': [None],                             
                }, index=[0]
                )
                df_base.loc[0,'landmarks'] = ps

                if not (df_traits == 0).any().any():
                    df = df.append(pd.concat([df_base, df_traits], axis=1),
                                    ignore_index=True)    
                #else:
                #    print(df_traits)
                #df = df.append(pd.concat([df_base, df_traits], axis=1),ignore_index=True)  
                
                
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


def addlabels(axes, i, j, x, y):
    for k in range(len(x)):
        axes[i,j].text(k+1, y[k], round(y[k],2), ha = 'center',
                       #bbox = dict(facecolor = 'red', alpha =.8)
                       )
       

if __name__ == "__main__":
    
    #with_types = True
    with_types = False

    t_k = traits_config.TRAITS_KEYS 
    
    root_data_directory = traits_config.ROOT_DATA_DIRECTORY
    statistics_file_name_suffix =""
    
    t_k_ex = traits_config.TRAITS_KEYS_EXCLUDED + traits_config.TRAITS_KEYS_SERVICE + traits_config.TRAITS_KEYS_AUX 
    
    if with_types:
        t_k.extend(['type'])
    else:
        t_k_ex.extend(['type'])
        
    print(t_k)
    print(t_k_ex)        
    
    tps_df=tps_list(traits_keys = t_k, traits_keys_excluded = t_k_ex, 
                    with_types = with_types,
                    root_data_directory = root_data_directory)    
    print(tps_df.head(31))
    print(tps_df.columns)
    
    ncols=2
    nrows=math.ceil(len(t_k)/ncols)
    
         
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    fig.tight_layout(pad=5.0)
    

    i = 0
    j = 0
    
    for trait in t_k:
        st, st_percents = statistics(tps_df, trait)
        print('Absolute counts:\n', st)
        print('In percents:\n',st_percents)
    
        
        axes[i,j].set_xlim(-1, (4,6)[trait=='type'])
        axes[i,j].set_ylim(0, 100)
        axes[i,j].set_title(f'{trait}')
        cat = list(st_percents.keys())
        val = list(st_percents.values())
        axes[i,j].bar(cat, val)
        addlabels(axes, i, j, cat, val)
        axes[i,j].grid()

        
        i += 1
        if i>=nrows:
            i=0
            j += 1
            if j>=ncols:
                j=0
                 
      
    plt.savefig(f"./outputs/dataset_statistics_{statistics_file_name_suffix}.png")
