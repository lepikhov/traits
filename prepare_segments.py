import os
import sys


import pandas as pd
from PIL import Image
import data_loading

import traits_config

sys.path.append('../keypoints-for-entire-image')
import predict



def merge_segments(merge_list, boxes, segments):
    
    x1, y1 = (sys.maxsize,)*2 
    x2, y2 = (0,)*2 
    
    for seg in merge_list:
        i=0
        for s in segments:
            if seg==s:
                break
            i += 1
            
        if i>=len(segments):
            return []
    
        box=boxes[i]
        if box[0]<x1:
            x1=box[0]
        
        if box[1]<y1:
            y1=box[1]
            
        if box[2]>x2:
            x2=box[2]
            
        if box[3]>y2:
            y2=box[3]        
              
    return [x1,y1,x2,y2]             


def copy_image_segments(image, segments, boxes, copy_segments_list, filename, src_df, dst_df, idx):
    
    boxes = merge_segments(copy_segments_list, boxes, segments)           
    
    if not boxes==[]:
        prfx_dir = traits_config.SEGMENTATION_DIRECTORY
        dir=''
        for s in copy_segments_list:
            if len(dir):
                dir += '_'
            dir += s.replace(' ','_') 
    
        if not os.path.exists(prfx_dir+'/'+dir):
            os.makedirs(prfx_dir+'/'+dir)
    
        im = image.crop(boxes)
        im.save(f'{prfx_dir}/{dir}/{filename}')       
        
        columns = dst_df.columns
        df = pd.DataFrame(columns=columns)
        df = df._append(src_df.iloc[idx][columns], ignore_index=True)
        df['imagefile'] = filename
        df['imagedir'] = prfx_dir+'/'+dir

        dst_df = dst_df.append(df, ignore_index=True)
        #print(dst_df)
        
    
    return dst_df
     
        
         




def prepare_segments():    

    df = data_loading.tps_list() 
    
    df_traits_head_neck = pd.DataFrame(columns=['id','imagedir','imagefile'] + traits_config.TRAITS_HEAD_NECK_KEYS)
    df_traits_head_neck_body = pd.DataFrame(columns=['id', 'imagedir','imagefile'] + traits_config.TRAITS_HEAD_NECK_BODY_KEYS)
    df_traits_rear_leg = pd.DataFrame(columns=['id', 'imagedir','imagefile'] + traits_config.TRAITS_REAR_LEG_KEYS)
    df_traits_front_leg = pd.DataFrame(columns=['id', 'imagedir','imagefile'] + traits_config.TRAITS_FRONT_LEG_KEYS)
    df_traits_body = pd.DataFrame(columns=['id', 'imagedir','imagefile'] + traits_config.TRAITS_BODY_KEYS)
    df_traits_body_front_leg = pd.DataFrame(columns=['id', 'imagedir','imagefile'] + traits_config.TRAITS_BODY_FRONT_LEG_KEYS)
    df_traits_body_neck = pd.DataFrame(columns=['id', 'imagedir', 'imagefile'] + traits_config.TRAITS_BODY_NECK_KEYS)    
    
    segmentation_model, _ = predict.prepare_models()
    
    for idx in range(len(df)):
        
        
        filepath = df.iloc[idx]['imagedir']        
        filename = df.iloc[idx]['imagefile']       
        image_path = os.path.join(filepath, filename)   

                 
        image = Image.open(image_path)   
        #image.save(f'./outputs/segmentation/prepare_segments_{filename}')
        
        
        boxes, segments, _, _  = predict.get_segments(segmentation_model, image)
        #print(boxes, segments)
        
        filename = str(idx)+'.png'
        df_traits_head_neck = copy_image_segments(image, segments, boxes, ['Head', 'Neck'], filename, 
                                                  df, df_traits_head_neck, idx)
        df_traits_head_neck_body = copy_image_segments(image, segments, boxes, ['Head', 'Neck', 'Body'], filename, 
                                                       df, df_traits_head_neck_body, idx)
        df_traits_rear_leg = copy_image_segments(image, segments, boxes, ['Rear leg'], filename, 
                                                 df, df_traits_rear_leg, idx)
        df_traits_front_leg = copy_image_segments(image, segments, boxes, ['Front leg'], filename, 
                                                  df, df_traits_front_leg, idx)
        df_traits_body = copy_image_segments(image, segments, boxes, ['Body', 'Front leg'], filename, 
                                             df, df_traits_body, idx)
        df_traits_body_front_leg = copy_image_segments(image, segments, boxes, ['Body', 'Neck'], filename, 
                                                       df, df_traits_body_front_leg, idx)
        df_traits_body_neck = copy_image_segments(image, segments, boxes, ['Body'], filename,
                                                   df, df_traits_body_neck, idx)
        if not idx%50:
            percents = idx/len(df)
            print(f'index: {idx}, ready: {percents:.2%}')        
                

    print(f'ready: 100.00%') 

    df_traits_head_neck.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_head_neck.json'), orient='table') 
    df_traits_head_neck_body.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_head_neck_body.json'), orient='table') 
    df_traits_rear_leg.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_rear_leg.json'), orient='table') 
    df_traits_front_leg.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_front_leg.json'), orient='table') 
    df_traits_body.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body.json'), orient='table') 
    df_traits_body_front_leg.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body_front_leg.json'), orient='table') 
    df_traits_body_neck.to_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body_neck.json'), orient='table') 



if __name__ == '__main__':    

    prepare_segments()
    
    #df_data = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body.json'), orient='table')
    #print(df_data)

    
    
    
        
        