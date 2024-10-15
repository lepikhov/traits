import os
import warnings

import torch
import torchvision.transforms as transforms
from dataset import AttributesDataset #, mean, std
from sklearn.metrics import balanced_accuracy_score
from config import TRAITS_KEYS, TRAITS_KEYS_MAP, DEVICE, models_weights
import torchvision.transforms as transforms


import cv2
import data_loading
import utils

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

    
    #sample = 234
    #sample = 432
    
    models=load_models(device=device, models_types=models_types)
    
    bad_idxs=[]
    #for idx in [100, 200, 300]:    
    for idx in range(len(df)):    
    #for idx in range(sample, sample+1): 
        filepath = df.iloc[idx]['imagedir']        
        filename = df.iloc[idx]['imagefile']       
        image_path = os.path.join(filepath, filename)    
        #print(image_path)
        image = cv2.imread(image_path)   
                   
        strong_err = 0
        weak_err = 0
        print('-' * 72)
        print(idx)
        
        res=predict(device=device, models=models, weights=models_weights, image=image)
        #print(res)
        
        for t in TRAITS_KEYS:
            real = df.iloc[idx][t]
                
            predicted=res[t][0][0][0]   
            #print(predicted) 
            strong_check = check_predict(int(real), predicted, weak=False)
            if not strong_check:
                strong_err += 1
                                            
            weak_check = check_predict(int(real), predicted, weak=True)
            if not weak_check:
                weak_err += 1
                    
            #print(t,':', int(real), predicted, strong_check, weak_check)
            #print(TRAITS_KEYS_MAP[t][0],':',TRAITS_KEYS_MAP[t][1][int(predicted)])                    
                    
            if weak_err > 3:
                bad_idxs.append(idx)
                                               

        print(f'number of strong wrong detected traits is: {strong_err}')                
        print(f'number of weak wrong detected traits is: {weak_err}')  

    print('*' * 72)                          
    if len(bad_idxs):
        print(f'very bad prediction for {bad_idxs} index(-es) of images')
    else:
        print('there are not too bad predictions')                