import os
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from dataset_segments import TraitsDataset, AttributesDataset
import pandas as pd

from models.mobilenet_segments import MultiOutputModel_Mobilenet
from models.resnet_segments import MultiOutputModel_Resnet
from models.squeezenet_segments import MultiOutputModel_Squeezenet
from models.harmonicnet_segments import MultiOutputModel_Harmonicnet
from models.efficientnet_segments import MultiOutputModel_Efficientnet
from models.vitnet_segments import MultiOutputModel_Vitnet

from test_segments import calculate_metrics, validate #, visualize_grid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import math



import traits_config 

# loss plots
def plot_loss(loss_list, model_type, color, loss_type, segments):
    plt.figure(figsize=(10, 7))
    plt.plot(loss_list, color=color, label=f'{loss_type} loss for {model_type} ({segments})')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    segments=segments.replace(' ','_')
    plt.savefig(f"./outputs/{segments}_{model_type}_{loss_type}_loss.png")
    
# accuracies plots
def plot_accuracies(accuracies_list, model_type, color, accuracy_type, segments, traits_keys):
    
    nrows = math.ceil(math.sqrt(len(traits_keys)))
    if nrows<2:
        nrows = 2
    ncols = math.ceil(len(traits_keys)/nrows)
    if ncols<2:
        ncols = 2
    
         
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3*nrows))
    fig.tight_layout(pad=5.0)
    

    i = 0
    j = 0
    
    for trait in traits_keys:
        
        val=[]
        for item in accuracies_list:
            val.append(item[trait])
        axes[i,j].set_title(f'{trait}')
        axes[i,j].set_xlabel('epochs')
        axes[i,j].set_ylabel('accuracy')
        axes[i,j].plot(val, color=color)
        axes[i,j].grid()
        
        
        i += 1
        if i>=nrows:
            i = 0
            j += 1
            if j>=ncols:
                j = 0
    
    segments=segments.replace(' ','_')
    plt.savefig(f"./outputs/{segments}_{model_type}_{accuracy_type}_accuracies.png")


def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)
    

if __name__ == '__main__':    

    start_epoch = 1
    N_epochs = 200
    batch_size = 16
    num_workers = 8  # number of processes to handle dataset loading
    
    #model_type = 'mobilenet'
    #model_type = 'resnet'
    model_type = 'squeezenet'
    #model_type = 'efficientnet'   
    #model_type = 'harmonicnet'     
    #model_type = 'vitnet' 
    
    segments = 'Head Neck'
    #segments = 'Head Neck Body'                                             
    #segments = 'Rear leg'                                                
    #segments = 'Front leg'                                                 
    #segments = 'Body'        
    #segments = 'Body Front leg'           
    #segments = 'Body Neck'

    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() and traits_config.DEVICE == 'cuda' else "cpu")

    print(traits_config.DEVICE, device)
    
    #torch.autograd.set_detect_anomaly(True)

    
    match segments:
        case 'Head Neck':     
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_head_neck.json'), orient='table') 
            traits_keys = traits_config.TRAITS_HEAD_NECK_KEYS          
        case 'Head Neck Body':                                                         
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_head_neck_body.json'), orient='table')
            traits_keys = traits_config.TRAITS_HEAD_NECK_BODY_KEYS
        case 'Rear leg':                                                 
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_rear_leg.json'), orient='table')
            traits_keys = traits_config.TRAITS_REAR_LEG_KEYS
        case 'Front leg':                                                  
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_front_leg.json'), orient='table')
            traits_keys = traits_config.TRAITS_FRONT_LEG_KEYS 
        case 'Body':        
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body.json'), orient='table')            
            traits_keys = traits_config.TRAITS_BODY_KEYS 
        case 'Body Front leg':             
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body_front_leg.json'), orient='table')
            traits_keys = traits_config.TRAITS_BODY_FRONT_LEG_KEYS 
        case 'Body Neck':                              
            df = pd.read_json(os.path.join(traits_config.SEGMENTATION_DIRECTORY,'df_traits_body_neck.json'), orient='table')
            traits_keys = traits_config.TRAITS_BODY_NECK_KEYS 
        case _:
            pass    
    
    
    
    
    
    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs    
    attributes = AttributesDataset(df, segments=segments)

    # specify image transforms for augmentation during training
    train_transform = A.Compose([
        #A.Resize(224, 224),        
        #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.OneOf([
            A.HueSaturationValue(p=0.5),
            A.RGBShift(p=0.7),           
        ], p=0.5),
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
            A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1),     
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.6, alpha_coef=0.1, p=1),    
        ], p=0.25),
        A.ToFloat(),
        ToTensorV2(),
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = A.Compose([
        #A.Resize(224, 224),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        A.ToFloat(),
        ToTensorV2(),        
    ])
    
    training_samples, valid_samples = model_selection.train_test_split(df, shuffle=True,
                                                                       random_state=None,
                                                                       test_size=0.2)    
    
    print('training_samples;', len(training_samples))
    print('valid_samples;', len(valid_samples))

    train_dataset = TraitsDataset(training_samples, attributes, traits_keys, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers )    

    val_dataset = TraitsDataset(valid_samples, attributes, traits_keys, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers ) 

         
    match model_type:
        case 'mobilenet':
            model = MultiOutputModel_Mobilenet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)
        case 'squeezenet':
            model = MultiOutputModel_Squeezenet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)
        case 'resnet':    
            model = MultiOutputModel_Resnet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)
        case 'efficientnet':    
            model = MultiOutputModel_Efficientnet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)   
        case 'harmonicnet':    
            model = MultiOutputModel_Harmonicnet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)         
        case 'vitnet':    
            model = MultiOutputModel_Vitnet(n_classes=attributes, pretrained=True, segments=segments, traits_keys=traits_keys).to(device)                               
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
    
    train_loss_list = []
    train_losses_list = []
    train_accuracies_list = []
    val_loss_list = []
    val_accuracies_list = []    

 
    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0
        accuracies = {}
        for t in traits_keys:
            accuracies[t] = 0


        for batch in train_dataloader:
            optimizer.zero_grad()

            img = batch['image']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracies = calculate_metrics(output, target_labels, traits_keys)

            for t in traits_keys:
                accuracies[t] += batch_accuracies[t]

            #loss_train.backward(retain_graph=True)
            loss_train.backward()
            optimizer.step()

        print(f"epoch {epoch}, loss: {total_loss / n_train_samples}")
        train_loss_list.append(total_loss / n_train_samples)
        
        acc = {}
        for t in traits_keys:
            print(f"{t}: {accuracies[t]/n_train_samples}", end='\t')
            acc[t]=accuracies[t]/n_train_samples
        print('\n')            
        
        train_accuracies_list.append(acc)
        

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        l, a = validate(model, val_dataloader, logger, epoch, device, traits_keys)
        val_loss_list.append(l)
        val_accuracies_list.append(a)        
        
        if epoch % 5 == 0:
            plot_loss(train_loss_list, model_type, 'orange', 'train', segments=segments)
            plot_accuracies(train_accuracies_list, model_type, 'green', 'train', segments=segments, traits_keys=traits_keys)
            plot_loss(val_loss_list, model_type, 'red', 'val', segments=segments)
            plot_accuracies(val_accuracies_list, model_type, 'blue', 'val', segments=segments, traits_keys=traits_keys)

        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)
            
            