#%%
import numpy as np
import pandas as pd
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import sys


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x

class PreModel(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        
        #PRETRAINED MODEL
        self.pretrained = models.resnet34(weights=None)
        
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        # resnet50 ProjectionHead(2048, 2048, 512)
        # resnet15 ProjectionHead(512, 512, 512)
        self.projector = ProjectionHead(512, 512, 512)  ## embedded size

    def forward(self,x):
        out = self.pretrained(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp


class DSModel(nn.Module):
    def __init__(self,premodel,num_classes):
        super().__init__()
        
        self.premodel = premodel
        self.num_classes = num_classes
        
        for p in self.premodel.parameters():
            p.requires_grad = True
            
        for p in self.premodel.projector.parameters():
            p.requires_grad = False
        
        # resnet50 ==> 2048
        self.lastlayer = nn.Linear(512,self.num_classes)
        
    def forward(self,x):
        out = self.premodel.pretrained(x)
        out = self.lastlayer(out)
        return out


#%%
# get all the image folder paths
all_paths = os.listdir('C:/CCBDA/HW2/unlabeled')
folder_paths = [x for x in all_paths if os.path.isdir('C:/CCBDA/HW2/unlabeled/' + x)]
folder_paths = [str(i) for i in folder_paths]
print(f"Folder paths: {folder_paths}")
print(f"Number of folders: {len(folder_paths)}")

#%%
# get the embedded representation before classification layer
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = PreModel('resnet18').to('cuda:0')
# optimizer = LARS(
#     [params for params in model.parameters() if params.requires_grad],
#     lr=0.2,
#     weight_decay=1e-6,
#     exclude_from_weight_decay=["batch_normalization", "bias"],
# )
# warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
# mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

checkpoint = torch.load('C:/CCBDA/HW2/model_weight/SimCLR_RN18_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_150_20221031.pt')
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# warmupscheduler.load_state_dict(checkpoint['scheduler_state_dict'])
# mainscheduler.load_state_dict(checkpoint['scheduler_state_dict'])
model.eval()


dsmodel = DSModel(model, 4).to('cuda:0')
# dsoptimizer = torch.optim.SGD([params for params in dsmodel.parameters() if params.requires_grad],lr = 0.01, momentum = 0.9)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose = True)

checkpoint = torch.load('C:/CCBDA/HW2/model_weight/rn18_p128_sgd0p01_decay0p98_all_lincls_20221031.pt')
dsmodel.load_state_dict(checkpoint['model_state_dict'])
# dsoptimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
dsmodel.eval()

#%%
from tqdm import tqdm
import sys

image_formats = ['jpg'] # we only want images that are in this format
result_arr = []

for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
    image_paths = os.listdir('C:/CCBDA/HW2/unlabeled/'+folder_path)
    label = folder_path
    # save image paths in the DataFrame
    for image_path in image_paths:
        if image_path.split('.')[-1] in image_formats:
            print(f"C:/CCBDA/HW2/unlabeled/{folder_path}/{image_path}")
            x = Image.open(f"C:/CCBDA/HW2/unlabeled/{folder_path}/{image_path}")
            x = x.convert('RGB')
            x = np.array(x)
            x = np.transpose(x, (2, 0, 1)).astype(np.float32)
            x = torch.from_numpy(x).float()
            # print(x.shape) # 3, 96, 96
            x = x.unsqueeze(0).to(device = 'cuda:0', dtype = torch.float)

            model.pretrained.fc.register_forward_hook(get_activation('pretrained.fc'))
            output = dsmodel(x)
            embedding = activation['pretrained.fc'].cpu().detach().numpy()
            # print(activation['pretrained.fc'])
            # print(type(activation['pretrained.fc']))
            # print(activation['pretrained.fc'].shape) # 1, 512 
            result_arr.append(np.squeeze(embedding, axis=0))

result_arr = np.stack(result_arr, axis=0)
print(result_arr.shape)

# %%
np.save('C:/CCBDA/HW2/310704014.npy', result_arr)

# %%
embeddings = np.load('C:/CCBDA/HW2/310704014.npy')
print(embeddings.dtype)
print(embeddings.shape)


# %%
