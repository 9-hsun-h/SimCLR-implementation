#%%
import numpy as np
import pandas as pd
import shutil, time, os, requests, random, copy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import sys

# might not be used.....

#%%
def set_seed(seed = 16):
    np.random.seed(seed)
    torch.manual_seed(seed)

class DataGenerator(Dataset):
    def __init__(self,phase,images,s = 0.5):
        self.phase = phase
        self.images = images
        self.s = s
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                              transforms.RandomResizedCrop(96,(0.8,1.0)),
                                              transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, 
                                                                                                                 0.8*self.s, 
                                                                                                                 0.8*self.s, 
                                                                                                                 0.2*self.s)], p = 0.8),
                                                                  transforms.RandomGrayscale(p=0.2)
                                                                 ])])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self,i):
        x = Image.open(self.images[i])
        x = x.convert('RGB')
        #print(x)
        #x = np.array(x)/255.0  # might not be needed
        x = np.array(x)  # might not be needed
        # print(x.shape)
        # print(f"x ====> {x}")
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        # print(x.shape)
        x1 = self.augment(torch.from_numpy(x))
        x2 = self.augment(torch.from_numpy(x))
        # print(f"x1 after augment and to_tensor {x1}")
        # print(x1.shape)
        # x1 = self.preprocess(x1)
        # x2 = self.preprocess(x2)
        return x1, x2

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self.images = self.images[random.sample(population = list(range(self.__len__())),k = self.__len__())]

    # def preprocess(self,frame):
    #     frame = (frame-MEAN)/STD
    #     return frame
    
    #applies randomly selected augmentations to each clip (same for each frame in the clip)
    def augment(self, frame, transformations = None):
        
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        
        return frame


#%%
# read the data.csv file and get the image paths and labels
df = pd.read_csv('C:/CCBDA/HW2/data.csv')
X = df.image_path.values # image paths
#y = df.target.values # targets
(x_train, x_valid) = train_test_split(X,
    test_size=0.10, random_state=42)
print(f"Training instances: {len(x_train)}")
print(f"Validation instances: {len(x_valid)}")

#%%
#create dataloader
train_dg = DataGenerator('train',x_train)
train_dl = DataLoader(train_dg,batch_size = 128,drop_last=True)

vavild_dg = DataGenerator('valid',x_valid)
valid_dl = DataLoader(vavild_dg,batch_size = 128,drop_last=True)
# %%
# for a,b in train_dl:
#     print(a.shape, b.shape)
#     break
# %%

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
        self.pretrained = models.resnet18(weights=None)
        
        self.pretrained.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.pretrained.maxpool = Identity()
        
        self.pretrained.fc = Identity()
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        # resnet50 ProjectionHead(2048, 2048, 512)
        # resnet18 ProjectionHead(512, 512, 512)
        self.projector = ProjectionHead(512, 512, 512)  ## embedded size

    def forward(self,x):
        out = self.pretrained(x)
        
        xp = self.projector(torch.squeeze(out))
        
        return xp

#%%
model = PreModel('resnet18').to('cuda:0')

#%%
# define nt_xent loss
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

def nt_xent(
    u: torch.Tensor,                               # [N, C]
    v: torch.Tensor,                               # [N, C]
    temperature: float = 0.5,
):
    """
    N: batch size
    C: feature dimension
    """
    N, C = u.shape

    z = torch.cat([u, v], dim=0)                   # [2N, C]
    z = F.normalize(z, p=2, dim=1)                 # [2N, C]
    s = torch.matmul(z, z.t()) / temperature       # [2N, 2N] similarity matrix
    mask = torch.eye(2 * N).bool().to(z.device)    # [2N, 2N] identity matrix
    s = torch.masked_fill(s, mask, -float('inf'))  # fill the diagonal with negative infinity
    label = torch.cat([                            # [2N]
        torch.arange(N, 2 * N),                    # {N, ..., 2N - 1}
        torch.arange(N),                           # {0, ..., N - 1}
    ]).to(z.device)

    loss = F.cross_entropy(s, label)               # NT-Xent loss
    return loss

#%%
from torch.optim.optimizer import Optimizer, required
import re

EETA_DEFAULT = 0.001


class LARS(Optimizer):
    """
    Layer-wise Adaptive Rate Scaling for large batch training.
    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.9,
        use_nesterov=False,
        weight_decay=0.0,
        exclude_from_weight_decay=None,
        exclude_from_layer_adaptation=None,
        classic_momentum=True,
        eeta=EETA_DEFAULT,
    ):
        """Constructs a LARSOptimizer.
        Args:
        lr: A `float` for learning rate.
        momentum: A `float` for momentum.
        use_nesterov: A 'Boolean' for whether to use nesterov momentum.
        weight_decay: A `float` for weight decay.
        exclude_from_weight_decay: A list of `string` for variable screening, if
            any of the string appears in a variable's name, the variable will be
            excluded for computing weight decay. For example, one could specify
            the list like ['batch_normalization', 'bias'] to exclude BN and bias
            from weight decay.
        exclude_from_layer_adaptation: Similar to exclude_from_weight_decay, but
            for layer adaptation. If it is None, it will be defaulted the same as
            exclude_from_weight_decay.
        classic_momentum: A `boolean` for whether to use classic (or popular)
            momentum. The learning rate is applied during momeuntum update in
            classic momentum, but after momentum for popular momentum.
        eeta: A `float` for scaling of learning rate when computing trust ratio.
        name: The name for the scope.
        """

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            exclude_from_weight_decay=exclude_from_weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eeta = eeta
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def step(self, epoch=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eeta = group["eeta"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: get param names
                # if self._use_weight_decay(param_name):
                grad += self.weight_decay * param

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: get param names
                    # if self._do_layer_adaptation(param_name):
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()
                    trust_ratio = torch.where(
                        w_norm.gt(0),
                        torch.where(
                            g_norm.gt(0),
                            (self.eeta * w_norm / g_norm),
                            torch.Tensor([1.0]).to(device),
                        ),
                        torch.Tensor([1.0]).to(device),
                    ).item()

                    scaled_lr = lr * trust_ratio
                    if "momentum_buffer" not in param_state:
                        next_v = param_state["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )
                    else:
                        next_v = param_state["momentum_buffer"]

                    next_v.mul_(momentum).add_(scaled_lr, grad)
                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)
                else:
                    raise NotImplementedError

        return loss

    def _use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

#%%
#OPTMIZER
optimizer = LARS(
    [params for params in model.parameters() if params.requires_grad],
    lr=0.2,
    weight_decay=1e-6,
    exclude_from_weight_decay=["batch_normalization", "bias"],
)

# "decay the learning rate with the cosine decay schedule without restarts"
#SCHEDULER OR LINEAR EWARMUP
warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)

#SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500, eta_min=0.05, last_epoch=-1, verbose = True)

#LOSS FUNCTION
criterion = SimCLR_Loss(batch_size = 64, temperature = 0.5)

#%%
# some extra functions
def save_model(model, optimizer, scheduler, current_epoch, name):
    out = os.path.join('C:/CCBDA/HW2/model_weight/',name.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict()}, out)

# def plot_features(model, num_classes, num_feats, batch_size):
#     preds = np.array([]).reshape((0,1))
#     gt = np.array([]).reshape((0,1))
#     feats = np.array([]).reshape((0,num_feats))
#     model.eval()
#     with torch.no_grad():
#         for x1,x2 in vdl:
#             x1 = x1.squeeze().to(device = 'cuda:0', dtype = torch.float)
#             out = model(x1)
#             out = out.cpu().data.numpy()#.reshape((1,-1))
#             feats = np.append(feats,out,axis = 0)
    
#     tsne = TSNE(n_components = 2, perplexity = 50)
#     x_feats = tsne.fit_transform(feats)
#     num_samples = int(batch_size*(valimages.shape[0]//batch_size))#(len(val_df)
    
#     for i in range(num_classes):
#         plt.scatter(x_feats[vallabels[:num_samples]==i,1],x_feats[vallabels[:num_samples]==i,0])
    
#     plt.legend([str(i) for i in range(num_classes)])
#     plt.show()

#%%
# Training===============================================================================
nr = 0
current_epoch = 0
epochs = 100
tr_loss = []
val_loss = []

for epoch in range(epochs):
        
    print(f"Epoch [{epoch}/{epochs}]\t")
    stime = time.time()

    model.train()
    tr_loss_epoch = 0
    
    for step, (x_i, x_j) in enumerate(train_dl):
        optimizer.zero_grad()
        x_i = x_i.squeeze().to('cuda:0').float()
        x_j = x_j.squeeze().to('cuda:0').float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        #loss = criterion(z_i, z_j)
        loss = nt_xent(z_i,z_j)
        loss.backward()

        optimizer.step()
        
        if nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_dl)}]\t Loss: {round(loss.item(), 5)}")

        tr_loss_epoch += loss.item()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()
    
    lr = optimizer.param_groups[0]["lr"]

    if nr == 0 and (epoch+1) % 50 == 0:
        save_model(model, optimizer, mainscheduler, current_epoch,"SimCLR_RN18_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_20221031.pt")

    model.eval()
    with torch.no_grad():
        val_loss_epoch = 0
        for step, (x_i, x_j) in enumerate(valid_dl):
        
          x_i = x_i.squeeze().to('cuda:0').float()
          x_j = x_j.squeeze().to('cuda:0').float()

          # positive pair, with encoding
          z_i = model(x_i)
          z_j = model(x_j)

          #loss = criterion(z_i, z_j)
          loss = nt_xent(z_i,z_j)

          if nr == 0 and step % 50 == 0:
              print(f"Step [{step}/{len(valid_dl)}]\t Loss: {round(loss.item(),5)}")

          val_loss_epoch += loss.item()

    if nr == 0:
        tr_loss.append(tr_loss_epoch / len(train_dl))
        val_loss.append(val_loss_epoch / len(valid_dl))
        print(f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(train_dl)}\t lr: {round(lr, 5)}")
        print(f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(valid_dl)}\t lr: {round(lr, 5)}")
        current_epoch += 1

    train_dg.on_epoch_end()

    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")

    # if (epoch+1)%10==0:
    #     plot_features(model.pretrained, 4, 2048, 128)

save_model(model, optimizer, mainscheduler, current_epoch, "SimCLR_RN18_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_20221031.pt")
#%%
plt.plot(tr_loss,'b-')
plt.plot(val_loss,'r-')
plt.legend(['t','v'])
plt.show()



# %% #downstream finetune model
class DSModel(nn.Module):
    def __init__(self,premodel,num_classes):
        super().__init__()
        
        self.premodel = premodel
        self.num_classes = num_classes
        
        for p in self.premodel.parameters():
            p.requires_grad = True
            
        for p in self.premodel.projector.parameters():
            p.requires_grad = False
        
        self.lastlayer = nn.Linear(512,self.num_classes)
        
    def forward(self,x):
        out = self.premodel.pretrained(x)
        out = self.lastlayer(out)
        return out
#%%
dsmodel = DSModel(model, 4).to('cuda:0')
#%%
class DSDataGen(Dataset):
    def __init__(self, phase, images,labels,num_classes):
        
        self.phase = phase
        self.num_classes = num_classes
        self.images = images
        self.labels = labels
        
        self.randomcrop = transforms.RandomResizedCrop(96,(0.8,1.0))
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self,i):
        
        x = Image.open(self.images[i])
        x = x.convert('RGB')
        x = np.array(x)
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(x).float()
        label = self.labels[i]

        if self.phase == 'train':
            img  = self.randomcrop(img)

        #img = self.preprocess(img)
        
        return img, torch.tensor(label, dtype=torch.long)
    
    def on_epoch_end(self):
        i = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.images = self.images[i]
        self.labels = self.labels[i]
        
    # def preprocess(self,frame):
    #     frame = frame / 255.0
    #     frame = (frame-MEAN)/STD
    #     return frame

#%% finetuning=============================================================================
# read the data_finetune.csv file and get the image paths and labels
df = pd.read_csv('C:/CCBDA/HW2/data_finetune.csv')
X = df.image_path.values # image paths
y = df.target.values # targets
(x_train, x_valid, y_train, y_valid) = train_test_split(X,y,
    test_size=0.10, random_state=42)
print(f"Training instances: {len(x_train)}")
print(f"Validation instances: {len(x_valid)}")
#%%
train_dg = DSDataGen('train', x_train, y_train, num_classes=4)

train_dl = DataLoader(train_dg,batch_size = 8, drop_last = True)

valid_dg = DSDataGen('valid', x_valid, y_valid, num_classes=4)

valid_dl = DataLoader(valid_dg,batch_size = 8, drop_last = True)

#%%
tr_ep_loss = []
tr_ep_acc = []

val_ep_loss = []
val_ep_acc = []

min_val_loss = 100.0

EPOCHS = 10
num_cl = 10

dsoptimizer = torch.optim.SGD([params for params in dsmodel.parameters() if params.requires_grad],lr = 0.001, momentum = 0.9)

lr_scheduler = torch.optim.lr_scheduler.StepLR(dsoptimizer, step_size=1, gamma=0.98, last_epoch=-1, verbose = True)

loss_fn = nn.CrossEntropyLoss()

#%%
for epoch in range(25):
    
    stime = time.time()
    print("=============== Epoch : %3d ==============="%(epoch+1))
    
    loss_sublist = np.array([])
    acc_sublist = np.array([])
    
    #iter_num = 0
    dsmodel.train()
    
    dsoptimizer.zero_grad()
    
    for x,y in train_dl:
        x = x.squeeze().to(device = 'cuda:0', dtype = torch.float)
        y = y.to(device = 'cuda:0')
        
        z = dsmodel(x)
        
        dsoptimizer.zero_grad()
        
        tr_loss = loss_fn(z,y)
        tr_loss.backward()

        preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))
        
        dsoptimizer.step()
        
        loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
        acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)
        
    print('ESTIMATING TRAINING METRICS.............')
    
    print('TRAINING BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
    print('TRAINING BINARY ACCURACY: ',np.mean(acc_sublist))
    
    tr_ep_loss.append(np.mean(loss_sublist))
    tr_ep_acc.append(np.mean(acc_sublist))
    
    print('ESTIMATING VALIDATION METRICS.............')
    
    dsmodel.eval()
    
    loss_sublist = np.array([])
    acc_sublist = np.array([])
    
    with torch.no_grad():
        for x,y in valid_dl:
            x = x.squeeze().to(device = 'cuda:0', dtype = torch.float)
            y = y.to(device = 'cuda:0')
            z = dsmodel(x)

            val_loss = loss_fn(z,y)

            preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))

            loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
            acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)
    
    print('VALIDATION BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
    print('VALIDATION BINARY ACCURACY: ',np.mean(acc_sublist))
    
    val_ep_loss.append(np.mean(loss_sublist))
    val_ep_acc.append(np.mean(acc_sublist))
    
    lr_scheduler.step()
    
    train_dg.on_epoch_end()
    
    if np.mean(loss_sublist) <= min_val_loss:
        min_val_loss = np.mean(loss_sublist) 
        print('Saving model...')
        torch.save({'model_state_dict': dsmodel.state_dict(),
                'optimizer_state_dict': dsoptimizer.state_dict()}, 
               'C:/CCBDA/HW2/model_weight/rn18_p128_sgd0p01_decay0p98_all_lincls_20221031.pt')
    
    print("Time Taken : %.2f minutes"%((time.time()-stime)/60.0))

#%%
plt.plot([t for t in tr_ep_acc])
plt.plot([t for t in val_ep_acc])
plt.legend(['train','valid'])

#%%
plt.plot(tr_ep_loss)
plt.plot(val_ep_loss)
plt.legend(['train','valid'])

#%% testing ============================================================================================
df = pd.read_csv('C:/CCBDA/HW2/data_test.csv')
X = df.image_path.values # image paths
y = df.target.values # targets
# (_, x_test, _, y_test) = train_test_split(X,y,
#     test_size=100, random_state=42)
x_test = X
y_test = y
print(f"Training instances: {len(x_test)}")
print(f"Validation instances: {len(y_test)}")
#%%
test_dg = DSDataGen('test', x_test, y_test, num_classes=4)

test_dl = DataLoader(test_dg, batch_size = 8, drop_last = True)

dsmodel.eval()
    
loss_sublist = np.array([])
acc_sublist = np.array([])

with torch.no_grad():
    for x,y in test_dl:
        x = x.squeeze().to(device = 'cuda:0', dtype = torch.float)
        y = y.to(device = 'cuda:0')
        z = dsmodel(x)

        val_loss = loss_fn(z,y)

        preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))

        loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
        acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)

print('TEST BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
print('TEST BINARY ACCURACY: ',np.mean(acc_sublist))




# %%
