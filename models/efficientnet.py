import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import TRAITS_KEYS


class MultiOutputModel_Efficientnet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()

            
        self.base_model = models.efficientnet_b3(pretrained=pretrained).features #take the model without classifier
        
        classifier = models.efficientnet_b3(pretrained=pretrained).classifier 

        last_channel = 1536  # size of the layer before classifier
    
        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # create separate classifiers for our outputs
        
        
        self.head_0 = classifier
        self.head_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_head_0)
        


        self.nape = classifier
        self.nape[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_nape)       

        self.neck_0 = classifier
        self.neck_0[1]  = nn.Linear(in_features=last_channel, out_features=n_classes.num_neck_0)


        self.withers_0 = classifier
        self.withers_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_0)


        self.shoulder = classifier
        self.shoulder[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_shoulder)


        self.spine_0 = classifier
        self.spine_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_0)         


        self.rump = classifier
        self.rump[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_rump)

        self.falserib_0 = classifier
        self.falserib_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_falserib_0)

        self.forearm = classifier
        self.forearm[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_forearm)
  

        self.headstock = classifier
        self.headstock[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_headstock)
             
        
        self.hip = classifier
        self.hip[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_hip)
            

        self.shin_0 = classifier
        self.shin_0[1]  = nn.Linear(in_features=last_channel, out_features=n_classes.num_shin_0)
     

        self.tailstock = classifier
        self.tailstock[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_tailstock)

        self.withers_1 = classifier
        self.withers_1[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_1)

        self.spine_3 = classifier
        self.spine_3[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_3)
 
        
        self.lower_back_0 = classifier
        self.lower_back_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_lower_back_0)
         

        self.rib_cage_0 = classifier
        self.rib_cage_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_rib_cage_0)
                      

        self.angle_4 = classifier
        self.angle_4[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)
               
        self.angle_5 = classifier
        self.angle_5[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_5)

        self.angle_10 = classifier
        self.angle_10[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_10)
        
        self.angle_11 = classifier
        self.angle_11[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_11)        
        
        self.angle_12 = classifier
        self.angle_12[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_12)       
        
        self.angle_13 = classifier
        self.angle_13[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_13)        
        
        self.angle_14 = classifier
        self.angle_14[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_14)        

        self.angle_15 = classifier
        self.angle_15[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_15)
      

    def forward(self, x):
        x = self.base_model(x)   
        
        
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)  
        
        # clone to avoid "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        y=[]
        for i in range(len(TRAITS_KEYS)):
            y.append(x.clone())      
                
        return {
            'head_0': self.head_0(y[0]),
            'nape': self.nape(y[1]), 
            'neck_0': self.neck_0(y[2]), 
            'withers_0': self.withers_0(y[3]), 
            'shoulder': self.shoulder(y[4]), 
            'spine_0': self.spine_0(y[5]), 
            'rump': self.rump(y[6]), 
            'falserib_0': self.falserib_0(y[7]), 
            'forearm': self.forearm(y[8]), 
            'headstock': self.headstock(y[9]), 
            'hip': self.hip(y[10]), 
            'shin_0': self.shin_0(y[11]), 
            'tailstock': self.tailstock(y[12]), 
            'withers_1': self.withers_1(y[13]), 
            'spine_3': self.spine_3(y[14]), 
            'lower_back_0': self.lower_back_0(y[15]), 
            'rib_cage_0': self.rib_cage_0(y[16]), 
            'angle_4': self.angle_4(y[17]), 
            'angle_5': self.angle_5(y[18]), 
            'angle_10': self.angle_10(y[19]), 
            'angle_11': self.angle_11(y[20]), 
            'angle_12': self.angle_12(y[21]), 
            'angle_13': self.angle_13(y[22]), 
            'angle_14': self.angle_14(y[23]), 
            'angle_15': self.angle_15(y[24])                 
        }


    def get_loss(self, net_output, ground_truth):
        losses={}
        total_loss=0
        for i in range(len(TRAITS_KEYS)):
            losses[TRAITS_KEYS[i]] = F.cross_entropy(net_output[TRAITS_KEYS[i]], ground_truth[TRAITS_KEYS[i]])
            total_loss += losses[TRAITS_KEYS[i]]
        return total_loss, losses
    