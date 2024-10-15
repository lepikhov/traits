import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from traits_config import TRAITS_KEYS


class MultiOutputModel_Mobilenet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()


        self.base_model = models.mobilenet_v2(pretrained=pretrained).features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier
        

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        self.head_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_head_0)
        )

        self.nape = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_nape)
        )        

        self.neck_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_neck_0)
        )

        self.withers_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_0)
        )        

        self.shoulder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_shoulder)
        )         

        self.spine_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_0)
        )      

        self.rump = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_rump)
        )           
        
        self.falserib_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_falserib_0)
        )                  

        self.forearm = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_forearm)
        )      

        self.headstock = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_headstock)
        )              
        
        self.hip = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_hip)
        )              

        self.shin_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_shin_0)
        )      

        self.tailstock = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_tailstock)
        )                  

        self.withers_1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_1)
        )                 

        self.spine_3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_3)
        )  
        
        self.lower_back_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_lower_back_0)
        )                        

        self.rib_cage_0 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_rib_cage_0)
        )        

        self.angle_4 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)
        )                

        self.angle_5 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_5)
        )     

        self.angle_10 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_10)
        )         

        self.angle_11 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_11)
        )               

        self.angle_12 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_12)
        )      

        self.angle_13 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_13)
        )                           
                
        self.angle_14 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_14)
        )       

        self.angle_15 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_15)
        )  
                     

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {
            'head_0': self.head_0(x), 
            'nape': self.nape(x), 
            'neck_0': self.neck_0(x), 
            'withers_0': self.withers_0(x), 
            'shoulder': self.shoulder(x), 
            'spine_0': self.spine_0(x), 
            'rump': self.rump(x), 
            'falserib_0': self.falserib_0(x), 
            'forearm': self.forearm(x), 
            'headstock': self.headstock(x), 
            'hip': self.hip(x), 
            'shin_0': self.shin_0(x), 
            'tailstock': self.tailstock(x), 
            'withers_1': self.withers_1(x), 
            'spine_3': self.spine_3(x), 
            'lower_back_0': self.lower_back_0(x), 
            'rib_cage_0': self.rib_cage_0(x), 
            'angle_4': self.angle_4(x), 
            'angle_5': self.angle_5(x), 
            'angle_10': self.angle_10(x), 
            'angle_11': self.angle_11(x), 
            'angle_12': self.angle_12(x), 
            'angle_13': self.angle_13(x), 
            'angle_14': self.angle_14(x), 
            'angle_15': self.angle_15(x)            
        }

    def get_loss(self, net_output, ground_truth):
        losses={}
        total_loss=0
        for t in TRAITS_KEYS:
            losses[t] = F.cross_entropy(net_output[t], ground_truth[t])
            total_loss += losses[t]
        return total_loss, losses