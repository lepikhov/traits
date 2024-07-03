import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset import TraitsDataset, AttributesDataset #, mean, std
from models.mobilenet import MultiOutputModel_Mobilenet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from torch.utils.data import DataLoader
from config import TRAITS_KEYS

def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch

def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        accuracies = {}
        for i in range(len(TRAITS_KEYS)):
            accuracies[TRAITS_KEYS[i]]=0

        for batch in dataloader:
            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracies = calculate_metrics(output, target_labels)

            for i in range(len(TRAITS_KEYS)):
                accuracies[TRAITS_KEYS[i]] += batch_accuracies[TRAITS_KEYS[i]]

    n_samples = len(dataloader)
    avg_loss /= n_samples
    for i in range(len(TRAITS_KEYS)):
        accuracies[TRAITS_KEYS[i]] /= n_samples
    print('-' * 72)
    print(f"Validation  loss: {avg_loss}\n")
    for i in range(len(TRAITS_KEYS)):
        print(f"{TRAITS_KEYS[i]}: {accuracies[TRAITS_KEYS[i]]}", end='\t')
    print('\n')        


    logger.add_scalar('val_loss', avg_loss, iteration)
    for i in range(len(TRAITS_KEYS)):
        logger.add_scalar(f'val_accuracy_{TRAITS_KEYS[i]}', accuracies[TRAITS_KEYS[i]], iteration)

    model.train()

def calculate_metrics(output, target):

    gts={}
    predicted={}
    for i in range(len(TRAITS_KEYS)):
        _, predicted[TRAITS_KEYS[i]] = output[TRAITS_KEYS[i]].cpu().max(1)
        gts[TRAITS_KEYS[i]] = target[TRAITS_KEYS[i]].cpu()

    accuracies={}
    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        for i in range(len(TRAITS_KEYS)):
            accuracies[TRAITS_KEYS[i]] = balanced_accuracy_score(y_true=gts[TRAITS_KEYS[i]].numpy(), y_pred=predicted[TRAITS_KEYS[i]].numpy())

    return accuracies    