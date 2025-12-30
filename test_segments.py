import os
import warnings

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset #, mean, std
from sklearn.metrics import balanced_accuracy_score
import traits_config 
import torchvision.transforms as transforms

from PIL import Image
import cv2
import data_loading

from traits_predict import checkpoint_load, check_predict, predict, models_types, load_models
from traits_predict_segments import predict_with_segments, predict_type, calculate_traits

import csv
import random

def validate(model, dataloader, logger, iteration, device, traits_keys, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracies = {}
        for t in traits_keys:
            accuracies[t]=0

        for batch in dataloader:
            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))            

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracies = calculate_metrics(output, target_labels, traits_keys)

            for t in traits_keys:
                accuracies[t] += batch_accuracies[t]

    n_samples = len(dataloader)
    avg_loss /= n_samples
    for t in traits_keys:
        accuracies[t] /= n_samples
    print('-' * 72)
    print(f"Validation  loss: {avg_loss}\n")
    for t in traits_keys:
        print(f"{t}: {accuracies[t]}", end='\t')
    print('\n')        

    if logger:
        logger.add_scalar('val_loss', avg_loss, iteration)
        for t in traits_keys:
            logger.add_scalar(f'val_accuracy_{t}', accuracies[t], iteration)

    model.train()
    
    return avg_loss, accuracies

def calculate_metrics(output, target, traits_keys):

    gts={}
    predicted={}
    for t in traits_keys:
        _, predicted[t] = output[t].cpu().max(1)
        gts[t] = target[t].cpu()

    accuracies={}
    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        for t in traits_keys:
            accuracies[t] = balanced_accuracy_score(y_true=gts[t].numpy(), y_pred=predicted[t].numpy())
            #accuracies[t] = accuracy_score(y_true=gts[t].numpy(), y_pred=predicted[t].numpy())

    return accuracies    



if __name__ == '__main__':   
    
    with_segments = True
    #with_segments = False   

    calculate = True
    #calculate = False  
    
    #t_k = traits_config.TRAITS_KEYS + traits_config.TRAITS_KEYS_AUX
    #root_data_directory = traits_config.ROOT_DATA_DIRECTORY_ORLOVSKAYA
    #df = data_loading.tps_list(t_k, root_data_directory=root_data_directory)
    
    if with_segments:
        with_types = True
        #with_types = False

        t_k = traits_config.TRAITS_KEYS 
        t_k_ex = traits_config.TRAITS_KEYS_EXCLUDED + traits_config.TRAITS_KEYS_SERVICE + traits_config.TRAITS_KEYS_AUX 
    
        if with_types:
            t_k.extend(['type'])
            root_data_directory = traits_config.ROOT_DATA_DIRECTORY_ORLOVSKAYA
        else:
            t_k_ex.extend(['type'])
            root_data_directory = traits_config.ROOT_DATA_DIRECTORY
        
        print(t_k)
        print(t_k_ex)        
    
        df=data_loading.tps_list(traits_keys = t_k, traits_keys_excluded = t_k_ex, 
                        with_types = with_types,
                        root_data_directory = root_data_directory,
                        ignore_empty = False)    
    else:

        t_k = traits_config.TRAITS_KEYS 
        t_k_ex = ['type', 'lower_back_len']
    
        root_data_directory = traits_config.ROOT_DATA_DIRECTORY_ORLOVSKAYA
        #root_data_directory = traits_config.ROOT_DATA_DIRECTORY
        
        print(t_k)
        print(t_k_ex)        
    
        df=data_loading.tps_list(traits_keys = t_k, traits_keys_excluded = t_k_ex, 
                        with_types = False,
                        root_data_directory = root_data_directory,
                        ignore_empty = False)           
        
    
    device = torch.device("cuda" if torch.cuda.is_available() and traits_config.DEVICE == 'cuda' else "cpu")

    
    WEAK_ERROR_WEIGHT = 0.6
    #WEAK_ERROR_WEIGHT = 1.0
    

    
    #sample = 234
    #sample = 432
    #sample = 22
    #sample = 1174
    #sample = 555
    #sample = 694
    #sample = 782
    #sample = 700
    #sample = 728
    #sample = 10
    
    number_of_samples = 100
    #breed = 'any'
    breed = 'orlovskaya'
    
    random.seed(42)
    indxes = random.sample(range(len(df)), number_of_samples)
    #indxes = [334]
    print('indexes: ',indxes)
    
    errors = {}
    for t in t_k:
        errors[t]=0 
    
    models=load_models(device=device, models_types=models_types)
    
    bad_idxs=[]
    counter=0
    log_list=[]
    
    #for idx in [100, 200, 300]:    
    #for idx in range(len(df)):  
    #for idx in range(0,1):  
    #for idx in range(sample, sample+2):
    for idx in indxes:     
        
        #if df.iloc[idx]['spine_3'] != 2:
        #    continue
          
        counter += 1 
        
        filepath = df.iloc[idx]['imagedir']        
        filename = df.iloc[idx]['imagefile']       
        image_path = os.path.join(filepath, filename)   
        
        #if (not 'ACHAL' in image_path):
        #    continue
                 
        #cv2.imwrite(f'./outputs/test_{filename}', image)
        
        log={'path': image_path.replace(root_data_directory,'')}
                   
        strong_err = 0
        weak_err = 0
        print('-' * 72)        
        print(idx)
        print(image_path)
        
        
        if calculate:
            try:
                image=Image.open(image_path) 
            except:
                continue  
            res=calculate_traits(device=device, image=image, breed=breed)      
            print(res)      
        elif with_segments:                      
            try:
                image=Image.open(image_path) 
            except:
                continue  
            res=predict_with_segments(device=device, image=image, breed=breed)
            if with_types:
                res_type=predict_type(device=device, image=image)
                res=res | res_type    
                print(res)              
        else:            
            try:
                image = cv2.imread(image_path) 
            except:
                continue                 
            res=predict(device=device, models=models, weights=traits_config.models_weights, image=image)        
            print(res)
        
        for t in t_k:
            real = df.iloc[idx][t]
                
            try:
                if calculate:
                    predicted=res[t]
                else:                    
                    predicted=res[t][2]   
            except:
                print(f'expection: no trait {t} in predicted')
                predicted=0                

            try:
                int(real)
            except:                 
                continue
                
            weak_check = check_predict(int(real), predicted, weak=True)
            if not weak_check: 
                weak_err += 1                               
                                            
            strong_check = check_predict(int(real), predicted, weak=False)
            if not strong_check:
                strong_err += 1
            
            if (not weak_check):
                errors[t] += 1.0
            else:
                if (not strong_check):
                    errors[t] += WEAK_ERROR_WEIGHT
                                   
                    
            print(t,':', int(real), predicted, strong_check, weak_check)
            print(traits_config.TRAITS_KEYS_MAP[t][0],':',traits_config.TRAITS_KEYS_MAP[t][1][int(predicted)])          
            
            log[f'{t} (predicted)']=predicted
            log[f'{t} (real)']=int(real)
                    
            if weak_err > 3:
                bad_idxs.append(idx)
                                               
        log['weak errors'] = weak_err
        log['strong errors'] = strong_err
        
        print(f'number of strong wrong detected traits is: {strong_err}')                
        print(f'number of weak wrong detected traits is: {weak_err}')  
        if (not (counter % 100)):
        #if (True):
            print(f'errors for {counter} samples:\n{errors}') 
            print(f'total sum of errors: {sum(errors.values())}')
        
        log_list.append(log)            

    print('*' * 72)                          
    if len(bad_idxs):
        print(f'very bad prediction for {bad_idxs} index(-es) of images')
    else:
        print('there are not too bad predictions')  
    print('*' * 72)         
        
    print(errors)        
    print(f'total sum of errors: {sum(errors.values())}')
    print('*' * 72) 
    accuracies={}
    for k, v in errors.items():
        accuracies[k] = (counter-errors[k])/counter

    accuracies = {k: v for k, v in sorted(accuracies.items(), key=lambda item: item[1])}
    
    print('<>' * 36)
    print(f'accuracies on {counter} samples (weak error weight {WEAK_ERROR_WEIGHT}):')        
    for k, v in accuracies.items():
        print(f'{k} : {traits_config.TRAITS_KEYS_MAP[k][0]} : {v}') 
    print(f'mean accuracy (weak error weight {WEAK_ERROR_WEIGHT}):', sum(accuracies.values())/len(traits_config.TRAITS_KEYS))  
    
    #write to csv
    csv_filename = 'outputs/test.csv'

    fieldnames = list(log_list[0].keys())
    
    
    with open(csv_filename, mode='w', newline='') as file:
        # Create a DictWriter object
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerows(log_list)
    
                 