import os
import warnings

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset #, mean, std
from sklearn.metrics import balanced_accuracy_score
from traits_config import TRAITS_KEYS, TRAITS_KEYS_MAP, DEVICE, models_weights
import torchvision.transforms as transforms


import cv2
import data_loading

from traits_predict import checkpoint_load, check_predict, predict, models_types, load_models


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracies = {}
        for t in TRAITS_KEYS:
            accuracies[t]=0

        for batch in dataloader:
            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracies = calculate_metrics(output, target_labels)

            for t in TRAITS_KEYS:
                accuracies[t] += batch_accuracies[t]

    n_samples = len(dataloader)
    avg_loss /= n_samples
    for t in TRAITS_KEYS:
        accuracies[t] /= n_samples
    print('-' * 72)
    print(f"Validation  loss: {avg_loss}\n")
    for t in TRAITS_KEYS:
        print(f"{t}: {accuracies[t]}", end='\t')
    print('\n')        


    logger.add_scalar('val_loss', avg_loss, iteration)
    for t in TRAITS_KEYS:
        logger.add_scalar(f'val_accuracy_{t}', accuracies[t], iteration)

    model.train()
    
    return avg_loss, accuracies

def calculate_metrics(output, target):

    gts={}
    predicted={}
    for t in TRAITS_KEYS:
        _, predicted[t] = output[t].cpu().max(1)
        gts[t] = target[t].cpu()

    accuracies={}
    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        for t in TRAITS_KEYS:
            accuracies[t] = balanced_accuracy_score(y_true=gts[t].numpy(), y_pred=predicted[t].numpy())
            #accuracies[t] = accuracy_score(y_true=gts[t].numpy(), y_pred=predicted[t].numpy())

    return accuracies    



if __name__ == '__main__':   
    
    df = data_loading.tps_list() 
    
    
    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE == 'cuda' else "cpu")

    
    WEAK_ERROR_WEIGHT = 0.7
    
    #sample = 234
    #sample = 432
    #sample = 22
    #sample = 1174
    #sample = 555
    #sample = 694
    #sample = 782
    sample = 700
    #sample = 728
    
    errors = {}
    for t in TRAITS_KEYS:
        errors[t]=0 
    
    models=load_models(device=device, models_types=models_types)
    
    bad_idxs=[]
    counter=0
    
    #for idx in [100, 200, 300]:    
    #for idx in range(len(df)):  
    #for idx in range(100):  
    for idx in range(sample, sample+1): 
        
        counter += 1 
        
        filepath = df.iloc[idx]['imagedir']        
        filename = df.iloc[idx]['imagefile']       
        image_path = os.path.join(filepath, filename)   
        
        #if (not 'ACHAL' in image_path):
        #    continue
                 
        image = cv2.imread(image_path)   
        cv2.imwrite(f'./outputs/test_{filename}', image)

                   
        strong_err = 0
        weak_err = 0
        print('-' * 72)        
        print(idx)
        print(image_path)
        
        res=predict(device=device, models=models, weights=models_weights, image=image)
        #print(res)
        
        for t in TRAITS_KEYS:
            real = df.iloc[idx][t]
                
            predicted=res[t][2]   

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
            print(TRAITS_KEYS_MAP[t][0],':',TRAITS_KEYS_MAP[t][1][int(predicted)])                    
                    
            if weak_err > 3:
                bad_idxs.append(idx)
                                               

        print(f'number of strong wrong detected traits is: {strong_err}')                
        print(f'number of weak wrong detected traits is: {weak_err}')  
        if (not (counter % 100)):
        #if (True):
            print(f'errors for {counter} samples:\n{errors}') 
            print(f'total sum of errors: {sum(errors.values())}')

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
        print(f'{k} : {TRAITS_KEYS_MAP[k][0]} : {v}') 
    print(f'mean accuracy (weak error weight {WEAK_ERROR_WEIGHT}):', sum(accuracies.values())/len(TRAITS_KEYS))        
                 