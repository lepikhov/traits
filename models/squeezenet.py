import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from config import TRAITS_KEYS


class MultiOutputModel_Squeezenet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()

            
        self.base_model = models.squeezenet1_1(pretrained=pretrained).features #take the model without classifier
        
        classifier = models.squeezenet1_1(pretrained=pretrained).classifier
        classifier.append(nn.Flatten()) #reshape [batch_size, num_classes, 1, 1] => [batch_size, num_classes]  
    

        # create separate classifiers for our outputs
        
        
        self.head_0 = classifier
        self.head_0[1] = nn.Conv2d(512, n_classes.num_head_0, kernel_size=(1,1), stride=(1,1))



        self.nape = classifier
        self.nape[1] = nn.Conv2d(512, n_classes.num_nape, kernel_size=(1,1), stride=(1,1))       


        self.neck_0 = classifier
        self.neck_0[1]  = nn.Conv2d(512, n_classes.num_neck_0, kernel_size=(1,1), stride=(1,1))


        self.withers_0 = classifier
        self.withers_0[1] = nn.Conv2d(512, n_classes.num_withers_0, kernel_size=(1,1), stride=(1,1))


        self.shoulder = classifier
        self.shoulder[1] = nn.Conv2d(512, n_classes.num_shoulder, kernel_size=(1,1), stride=(1,1))


        self.spine_0 = classifier
        self.spine_0[1] = nn.Conv2d(512, n_classes.num_spine_0, kernel_size=(1,1), stride=(1,1))          


        self.rump = classifier
        self.rump[1] = nn.Conv2d(512, n_classes.num_rump, kernel_size=(1,1), stride=(1,1))

        self.falserib_0 = classifier
        self.falserib_0[1] = nn.Conv2d(512, n_classes.num_falserib_0, kernel_size=(1,1), stride=(1,1))

        self.forearm = classifier
        self.forearm[1] = nn.Conv2d(512, n_classes.num_forearm, kernel_size=(1,1), stride=(1,1))
  

        self.headstock = classifier
        self.headstock[1] = nn.Conv2d(512, n_classes.num_headstock, kernel_size=(1,1), stride=(1,1))
             
        
        self.hip = classifier
        self.hip[1] = nn.Conv2d(512, n_classes.num_hip, kernel_size=(1,1), stride=(1,1))
            

        self.shin_0 = classifier
        self.shin_0[1]  = nn.Conv2d(512, n_classes.num_shin_0, kernel_size=(1,1), stride=(1,1))
     

        self.tailstock = classifier
        self.tailstock[1] = nn.Conv2d(512, n_classes.num_tailstock, kernel_size=(1,1), stride=(1,1))

        self.withers_1 = classifier
        self.withers_1[1] = nn.Conv2d(512, n_classes.num_withers_1, kernel_size=(1,1), stride=(1,1))

        self.spine_3 = classifier
        self.spine_3[1] = nn.Conv2d(512, n_classes.num_spine_3, kernel_size=(1,1), stride=(1,1))
 
        
        self.lower_back_0 = classifier
        self.lower_back_0[1] = nn.Conv2d(512, n_classes.num_lower_back_0, kernel_size=(1,1), stride=(1,1))
         

        self.rib_cage_0 = classifier
        self.rib_cage_0[1] = nn.Conv2d(512, n_classes.num_rib_cage_0, kernel_size=(1,1), stride=(1,1))
                      

        self.angle_4 = classifier
        self.angle_4[1] = nn.Conv2d(512, n_classes.num_angle_4, kernel_size=(1,1), stride=(1,1))
               
        self.angle_5 = classifier
        self.angle_5[1] = nn.Conv2d(512, n_classes.num_angle_5, kernel_size=(1,1), stride=(1,1))

        self.angle_10 = classifier
        self.angle_10[1] = nn.Conv2d(512, n_classes.num_angle_10, kernel_size=(1,1), stride=(1,1))
        
        self.angle_11 = classifier
        self.angle_11[1] = nn.Conv2d(512, n_classes.num_angle_11, kernel_size=(1,1), stride=(1,1))        
        
        self.angle_12 = classifier
        self.angle_12[1] = nn.Conv2d(512, n_classes.num_angle_12, kernel_size=(1,1), stride=(1,1))        
        
        self.angle_13 = classifier
        self.angle_13[1] = nn.Conv2d(512, n_classes.num_angle_13, kernel_size=(1,1), stride=(1,1))        
        
        self.angle_14 = classifier
        self.angle_14[1] = nn.Conv2d(512, n_classes.num_angle_14, kernel_size=(1,1), stride=(1,1))        

        self.angle_15 = classifier
        self.angle_15[1] = nn.Conv2d(512, n_classes.num_angle_15, kernel_size=(1,1), stride=(1,1))
           

    def forward(self, x):
        x = self.base_model(x)   
                
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
        for i in range(len(TRAITS_KEYS)):
            losses[TRAITS_KEYS[i]] = F.cross_entropy(net_output[TRAITS_KEYS[i]], ground_truth[TRAITS_KEYS[i]])
            total_loss += losses[TRAITS_KEYS[i]]
        return total_loss, losses