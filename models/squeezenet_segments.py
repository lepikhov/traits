import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class MultiOutputModel_Squeezenet(nn.Module):
    def __init__(self, n_classes, pretrained=True, segments_type='', traits_keys=None):
        super().__init__()

            
        self.base_model = models.squeezenet1_1(pretrained=pretrained).features #take the model without classifier
        
        classifier = models.squeezenet1_1(pretrained=pretrained).classifier
        classifier.append(nn.Flatten()) #reshape [batch_size, num_classes, 1, 1] => [batch_size, num_classes]  
    
        self.traits_keys = traits_keys
        self.segments_type = segments_type

        # create separate classifiers for our outputs
        match segments_type:
            case 'Type':    
                
                self.type_expressiveness = classifier                
                self.type_expressiveness[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_type_expressiveness, kernel_size=(1,1), stride=(1,1))                              
                )   
                            
            case 'Head Neck':    
                
                self.nape = classifier                
                self.nape[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_nape, kernel_size=(1,1), stride=(1,1))                              
                )                    
            
            case 'Head Neck Body':    
        
                self.head_0 = classifier
                self.head_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_head_0, kernel_size=(1,1), stride=(1,1))
                )                    

                self.withers_0 = classifier
                self.withers_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_withers_0, kernel_size=(1,1), stride=(1,1))
                )
                
                self.spine_0 = classifier
                self.spine_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_spine_0, kernel_size=(1,1), stride=(1,1))   
                )                    

                self.withers_1 = classifier
                self.withers_1[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_withers_1, kernel_size=(1,1), stride=(1,1))
                )                    

                self.rib_cage_0 = classifier
                self.rib_cage_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_rib_cage_0, kernel_size=(1,1), stride=(1,1))    
                )                    

                self.angle_4 = classifier
                self.angle_4[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_4, kernel_size=(1,1), stride=(1,1))        
                )                    
        
            case 'Rear leg':
        
                self.shin_0 = classifier
                self.shin_0[1]  = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_shin_0, kernel_size=(1,1), stride=(1,1))
                )                    
     
                self.tailstock = classifier
                self.tailstock[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_tailstock, kernel_size=(1,1), stride=(1,1))
                )                    
        
                self.angle_13 = classifier
                self.angle_13[1] = nn.Sequential(
                    nn.Dropout(p=0.2), 
                    nn.Conv2d(512, n_classes.num_angle_13, kernel_size=(1,1), stride=(1,1))              
                )                    

                self.angle_15 = classifier
                self.angle_15[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_15, kernel_size=(1,1), stride=(1,1))
                )                    
                
            case 'Front leg':   
                
                self.headstock = classifier
                self.headstock[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_headstock, kernel_size=(1,1), stride=(1,1))   
                )                    
                
                self.angle_12 = classifier
                self.angle_12[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_12, kernel_size=(1,1), stride=(1,1))            
                )                    
        
            case 'Body': 
                
                self.rump = classifier
                self.rump[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_rump, kernel_size=(1,1), stride=(1,1))
                )                    
                
                self.spine_3 = classifier
                self.spine_3[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_spine_3, kernel_size=(1,1), stride=(1,1))
                )                    
                
                self.lower_back_0 = classifier
                self.lower_back_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_lower_back_0, kernel_size=(1,1), stride=(1,1))
                )                                        

                self.angle_10 = classifier
                self.angle_10[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_10, kernel_size=(1,1), stride=(1,1))                
                )                    
                
            case 'Body Front leg': 
                
                self.shoulder = classifier
                self.shoulder[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_shoulder, kernel_size=(1,1), stride=(1,1))
                )                    

                self.falserib_0 = classifier
                self.falserib_0[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_falserib_0, kernel_size=(1,1), stride=(1,1))
                )

                self.forearm = classifier
                self.forearm[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_forearm, kernel_size=(1,1), stride=(1,1))
                )
                
                self.angle_5 = classifier
                self.angle_5[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_5, kernel_size=(1,1), stride=(1,1))      
                )
                
            case 'Body Neck':    
                
                self.neck_0 = classifier
                self.neck_0[1]  = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_neck_0, kernel_size=(1,1), stride=(1,1))
                )
                
                self.angle_4 = classifier
                self.angle_4[1] = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Conv2d(512, n_classes.num_angle_4, kernel_size=(1,1), stride=(1,1))                    
                )
                                
            case _:
                pass                                             

    def forward(self, x):
        x = self.base_model(x)   
                
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
            losses[t] = F.cross_entropy(net_output[t], ground_truth[t], ignore_index=0)
            total_loss += losses[t]
        return total_loss, losses