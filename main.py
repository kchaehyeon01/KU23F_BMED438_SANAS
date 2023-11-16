import os
import math
import argparse
import torch
import torch.utils.tensorboard
import torchsummary
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import * 
from utils.data import *
from utils.show_point_cloud import *

from models.basic_mlp import *


transform = None

train_dset = ShapeNetCore(
    path='./data/shapenet.hdf5', # 원래 : args.dataset_path, # './data/shapenet.hdf5'
    cates=['jar','monitor','chair','table','airplane'], # 원래 : args.categories, # default=['airplane']
    split='train',
    scale_mode='shape_unit',  # 원래 :  args.scale_mode,  # default='shape_unit'
    transform=transform,
)

train_loader = DataLoader(
    train_dset,
    batch_size=128, # 원래 : args.train_batch_size,
    num_workers=0,
)

model = basic_MLP()
for batch in train_loader:
    batch_pc = batch['pointcloud']
    result = model(batch_pc)


# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# torchsummary.summary(model,(128,2048,3))





print("for debug")

# check1 = next(train_iter)
# jar1 = check1['pointcloud'][0]
# show_point_cloud(torch.Tensor.numpy(jar1),0)


# pco = train_dset.__getitem__(1000)['pointcloud']
# # rotate 45 degree along x axis
# rmat = torch.Tensor([[1,0,0],[0,np.sqrt(2)/2,-np.sqrt(2)/2],[0,np.sqrt(2)/2,np.sqrt(2)/2]])
# pcr = torch.matmul(pco,rmat)
# show_point_cloud(torch.Tensor.numpy(pco),0)
# show_point_cloud(torch.Tensor.numpy(pcr),0)