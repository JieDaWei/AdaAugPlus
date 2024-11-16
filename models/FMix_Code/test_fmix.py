import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import fmix
import torch
import numpy as np
from tqdm import tqdm
import random

trainset = torchvision.datasets.ImageFolder(root='../CEN/',
                                            transform=transforms.ToTensor()) 
bs = 12
train_dataloader = DataLoader(trainset, batch_size=bs)
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
num = 0
for i,data in enumerate(train_dataloader):
    length = bs*len(train_dataloader)
    inputs, labels = data
    # print(inputs.shape)
    # print(labels)
    
    for j in range(0,12):
        filename = str(trainset.imgs[num])


        k = 0 
        knum = 0
        for substr in filename:
            k += 1
            if(substr=='/'):
                knum += 1
            if(knum==3):
                break
    
        subdir = filename[9:k-1]
        print(subdir)
        filename = filename[k:len(filename)-8]
        print(filename)


        num += 1
        print(str(num)+'/'+str(length))
        lam, mask = fmix.sample_mask(0.5, 3, [256,256,1], 0.3, True)
        index = random.randint(0,11)
        # print(inputs[j].shape)
        # print(mask.shape)
        mask = mask[0]
        # inputs[j] = inputs[j].transpose(1,2,0)
        mask = mask.transpose(2,0,1)
        x1, x2 = inputs[j] * mask, inputs[index] * (1-mask)
        # torchvision.utils.save_image(x1,'../Data_FMix_BCD/mixone{}.png'.format(j))
        # torchvision.utils.save_image(x2,'../Data_FMix_BCD/mixtwo{}.png'.format(j))
        mix = x1 + x2
        # print('../Data_FMix_BCD/{}/{}png'.format(subdir,filename))
        torchvision.utils.save_image(mix,'../Data_FMix_BCD/{}/{}png'.format(subdir,filename))
    