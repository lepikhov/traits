import os
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset import TraitsDataset, AttributesDataset
from models.mobilenet import MultiOutputModel_Mobilenet
from models.resnet import MultiOutputModel_Resnet
from test import calculate_metrics, validate#, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



import data_loading
from config import TRAITS_KEYS, DEVICE

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)

if __name__ == '__main__':    

    start_epoch = 1
    N_epochs = 50
    batch_size = 16
    num_workers = 8  # number of processes to handle dataset loading


    device = torch.device("cuda" if torch.cuda.is_available() and DEVICE == 'cuda' else "cpu")

    print(DEVICE, device)

    df = data_loading.tps_list() 
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(df)

    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        #transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
        #                        shear=None, resample=False, fill=(255, 255, 255)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    train_dataset = TraitsDataset(df, attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)    

    val_dataset = TraitsDataset(df, attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)   

    model_type = 'mobilenet'
    #model_type = 'resnet'
    match model_type:
        case 'mobilenet':
            model = MultiOutputModel_Mobilenet(n_classes=attributes).to(device)
        case 'squeezenet':
            pass
        case 'resnet':    
            model = MultiOutputModel_Resnet(n_classes=attributes).to(device)
        case _:
            pass  
     

    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join('./logs/', f'{model_type}-{get_cur_time()}')
    savedir = os.path.join('./checkpoints/', f'{model_type}-{get_cur_time()}')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)


    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracies = {}
        for i in range(len(TRAITS_KEYS)):
            accuracies[TRAITS_KEYS[i]] = 0


        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracies = calculate_metrics(output, target_labels)

            for i in range(len(TRAITS_KEYS)):
                accuracies[TRAITS_KEYS[i]] += batch_accuracies[TRAITS_KEYS[i]]

            loss_train.backward()
            optimizer.step()

        print(f"epoch {epoch}, loss: {total_loss / n_train_samples}")
        for i in range(len(TRAITS_KEYS)):
            print(f"{TRAITS_KEYS[i]}: {accuracies[TRAITS_KEYS[i]]/n_train_samples}", end='\t')
        print('\n')            

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, device)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)