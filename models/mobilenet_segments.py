import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
#from focal_loss.focal_loss import FocalLoss




class MultiOutputModel_Mobilenet(nn.Module):
    def __init__(self, n_classes, pretrained=True, segments_type='', traits_keys=None):
        super().__init__()


        self.base_model = models.mobilenet_v2(pretrained=pretrained).features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before classifier
        

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.traits_keys = traits_keys
        self.segments_type = segments_type

        # create separate classifiers for our outputs

        match segments_type:
            case 'Type':
                
                self.type_expressiveness = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_type_expressiveness)
                )        
                        
            case 'Head Neck':    
                
                self.nape = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_nape)
                )       
                       
            case 'Head Neck Body': 
                
                self.head_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_head_0)
                )   
                
                self.withers_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_0)
                )    
                
                self.spine_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_0)
                )                 

                self.withers_1 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_1)
                )  
                
                self.rib_cage_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_rib_cage_0)
                )         
                
                self.angle_4 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)
                )                         
                                                                    
            case 'Rear leg':     
                                
                self.shin_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_shin_0)
                )       
                
                self.tailstock = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_tailstock)
                )           
                
                self.angle_13 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_13)
                )                                                        

                self.angle_15 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_15)
                )                                       
                                                        
            case 'Front leg':        
                
                self.headstock = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_headstock)
                )   
                             
                self.angle_12 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_12)
                )                      
                                                                           
            case 'Body':   
                
                self.rump = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_rump)
                )   

                self.spine_3 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_3)
                )  
                
                self.lower_back_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_lower_back_0)
                )       
                
                self.angle_10 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_10)
                )                          
                                 
            case 'Body Front leg':    
                
                self.shoulder = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_shoulder)
                )         

                self.falserib_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_falserib_0)
                )                  

                self.forearm = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_forearm)
                )      
                    

                self.angle_5 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_5)
                )       
                                   
            case 'Body Neck':      
                
                self.neck_0 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_neck_0)
                )  
                
                self.angle_4 = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)
                )  
                                                                           
            case _:
                pass          
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)
        
        match self.segments_type:
            case 'Type':
                return {
                    'type' : self.type_expressiveness(x),
                }                
            case 'Head Neck': 
                return {
                    'nape': self.nape(x),                     
                }
            case 'Head Neck Body':                                                    
                return {
                    'head_0': self.head_0(x), 
                    'withers_0': self.withers_0(x), 
                    'spine_0': self.spine_0(x), 
                    'withers_1': self.withers_1(x),
                    'rib_cage_0': self.rib_cage_0(x),
                    'angle_4': self.angle_4(x), 
                }
            case 'Rear leg':                                    
                return {
                    'shin_0': self.shin_0(x), 
                    'tailstock': self.tailstock(x), 
                    'angle_13': self.angle_13(x), 
                    'angle_15': self.angle_15(x),                                          
                }
            case 'Front leg':                  
                return {
                    'headstock': self.headstock(x),
                    'angle_12': self.angle_12(x),                                         
                }
            case 'Body':                   
                return {
                    'rump': self.rump(x), 
                    'spine_3': self.spine_3(x), 
                    'lower_back_0': self.lower_back_0(x),
                    'angle_10': self.angle_10(x),
                }
            case 'Body Front leg':                 
                return {
                    'shoulder': self.shoulder(x),             
                    'falserib_0': self.falserib_0(x), 
                    'forearm': self.forearm(x), 
                    'angle_5': self.angle_5(x),                                         
                }
            case 'Body Neck':                   
                return {
                    'neck_0': self.neck_0(x),
                    'angle_4': self.angle_4(x), 
                }
            case _:
                return {}
                    

    def get_loss(self, net_output, ground_truth):
        losses={}
        total_loss=0
                   
        for t in self.traits_keys:                             
                #weights = torch.FloatTensor([0.01, 1/0.14, 1/0.63, 1/0.23]).to(ground_truth[t].device)
                #weights = torch.FloatTensor([0.01, 1/0.0627, 1/0.4495, 1/0.2578, 1/0.1672, 1/0.0627]).to(ground_truth[t].device)
                #m = torch.nn.Softmax(dim=-1)
                #losses[t] = FocalLoss(gamma=0.7, weights = weights)(m((net_output[t])), ground_truth[t])
                #losses[t] = F.cross_entropy(m((net_output[t])), ground_truth[t]) 
                losses[t] = F.cross_entropy(net_output[t], ground_truth[t], ignore_index=0)                
                total_loss += losses[t]
        return total_loss, losses