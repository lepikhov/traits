import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel_Efficientnet(nn.Module):
    def __init__(self, n_classes, pretrained=True, segments_type='', traits_keys=None):
        super().__init__()

            
        self.base_model = models.efficientnet_b3(pretrained=pretrained).features #take the model without classifier
        
        classifier = models.efficientnet_b3(pretrained=pretrained).classifier 

        last_channel = 1536  # size of the layer before classifier
    
        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.traits_keys = traits_keys
        self.segments_type = segments_type        

        # create separate classifiers for our outputs
        
        match segments_type:
            case 'Type':        
                
                self.type_expressiveness = classifier
                self.type_expressiveness[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_type_expressiveness)   
                           
            case 'Head Neck':        
                
                self.nape = classifier
                self.nape[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_nape)   

            case 'Head Neck Body':               
        
                self.head_0 = classifier
                self.head_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_head_0)
                
                self.withers_0 = classifier
                self.withers_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_0)
                
                self.spine_0 = classifier
                self.spine_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_0) 
                
                self.withers_1 = classifier
                self.withers_1[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_withers_1)
                
                self.rib_cage_0 = classifier
                self.rib_cage_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_rib_cage_0)
                
                self.angle_4 = classifier
                self.angle_4[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)
            
            case 'Rear leg': 
                
                self.shin_0 = classifier
                self.shin_0[1]  = nn.Linear(in_features=last_channel, out_features=n_classes.num_shin_0)
                
                self.tailstock = classifier
                self.tailstock[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_tailstock)
                
                self.angle_13 = classifier
                self.angle_13[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_13) 
                
                self.angle_15 = classifier
                self.angle_15[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_15)
                
            case 'Front leg': 
                
                self.headstock = classifier
                self.headstock[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_headstock)
                
                self.angle_12 = classifier
                self.angle_12[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_12) 
                
            case 'Body': 
                
                self.rump = classifier
                self.rump[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_rump)        
                
                self.spine_3 = classifier
                self.spine_3[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_spine_3)
 
        
                self.lower_back_0 = classifier
                self.lower_back_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_lower_back_0)  
                
                self.angle_10 = classifier
                self.angle_10[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_10)                      
                
            case 'Body Front leg':    
                
                self.shoulder = classifier
                self.shoulder[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_shoulder)   
                
                self.falserib_0 = classifier
                self.falserib_0[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_falserib_0)

                self.forearm = classifier
                self.forearm[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_forearm)     
                
                self.angle_5 = classifier
                self.angle_5[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_5)                     
                
            case 'Body Neck':                
        
                self.neck_0 = classifier
                self.neck_0[1]  = nn.Linear(in_features=last_channel, out_features=n_classes.num_neck_0)
                
                self.angle_4 = classifier
                self.angle_4[1] = nn.Linear(in_features=last_channel, out_features=n_classes.num_angle_4)  
                
            case _:
                pass                      
      

    def forward(self, x):
        x = self.base_model(x)   
        
        
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)  
        
       
        y=[]
        # clone to avoid "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
        for i in range(len(self.traits_keys)):
            y.append(x.clone())           
        
        match self.segments_type:
            case 'Type':                                    
                return {
                    'type': self.type_expressiveness(y[0]), 
                } 
                            
            case 'Head Neck':                                    
                return {
                    'nape': self.nape(y[0]), 
                }   
                
            case 'Head Neck Body':       
                return {
                    'head_0': self.head_0(y[0]),
                    'withers_0': self.withers_0(y[1]), 
                    'spine_0': self.spine_0(y[2]), 
                    'withers_1': self.withers_1(y[3]), 
                    'rib_cage_0': self.rib_cage_0(y[4]), 
                    'angle_4': self.angle_4(y[5]), 
                }    
                
            case 'Rear leg':          
                return {
                    'shin_0': self.shin_0(y[0]), 
                    'tailstock': self.tailstock(y[1]), 
                    'angle_13': self.angle_13(y[2]), 
                    'angle_15': self.angle_15(y[3])   
                }   
                
            case 'Front leg':         
                return { 
                    'headstock': self.headstock(y[0]), 
                    'angle_12': self.angle_12(y[1]), 
                }   
                
            case 'Body':      
                return { 
                    'rump': self.rump(y[0]),                         
                    'spine_3': self.spine_3(y[1]),
                    'lower_back_0': self.lower_back_0(y[2]), 
                    'angle_10': self.angle_10(y[3]), 
                } 
                
            case 'Body Front leg':             
                return { 
                    'shoulder': self.shoulder(y[0]),            
                    'falserib_0': self.falserib_0(y[1]), 
                    'forearm': self.forearm(y[2]), 
                    'angle_5': self.angle_5(y[3]), 
                }   
                
            case 'Body Neck':                                         
                return { 
                    'neck_0': self.neck_0(y[0]),
                    'angle_4': self.angle_4(y[1]), 
                }        
                                                                                                                       
            case _:
                return {}                


    def get_loss(self, net_output, ground_truth):
        losses={}
        total_loss=0
        for t in self.traits_keys:
            losses[t] = F.cross_entropy(net_output[t], ground_truth[t])
            total_loss += losses[t]
        return total_loss, losses
    