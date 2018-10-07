import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold , KFold
import torch.nn.functional as F
from fastai import core

class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,128)
        self.up2 = UnetBlock(128,128,128)
        self.up3 = UnetBlock(128,64,128)
        self.up4 = UnetBlock(128,64,128)
        self.up5 = nn.ConvTranspose2d(128, 1, 2, stride=2)
        
    def forward(self,img,depth):
        x = F.relu(self.rn(img))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()
            
class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
    
class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))
    
class UnetModel():
    def __init__(self,model,lr_cut,name='unet'):
        self.model,self.name = model,name
        self.lr_cut = lr_cut

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [self.lr_cut]))
        return lgs + [children(self.model)[1:]]
