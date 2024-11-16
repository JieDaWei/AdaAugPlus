import numpy as np
import matplotlib.pyplot as plt
import os
from torch.distributions import Categorical
import utils
from models.networks import *
import torch
import torch.optim as optim
from models.a2c import ActorCriticV5
import torchvision.models as tmodels
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from skimage import util, img_as_float, io
from misc.logger_tool import Logger, Timer
from models import augmentations,cutout,re_aug
from models.grid import *
from utils import de_norm
import time
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import random
import cv2

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def CutOut(images,images2,labels):
    cutout_aug = cutout.Cutout(n_holes=1, length=128)
    
    B,_,_,_ = images.shape
    # print(B)
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    # spath = "/home/wrf/4TDisk/CD/SG/results/retrain/ADCD/Data_Aug_Op/"
    for i in range(B):
        
        img1_aug,img2_aug,label_aug = cutout_aug(images[i],images2[i],labels[i])
        aug1.append(img1_aug)
        aug2.append(img2_aug)
        aug3.append(label_aug)
        # break

    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return cg1, cg2, cg3

def CutOut2(images,images2,labels,i):
    cutout_aug = cutout.Cutout(n_holes=1, length=128)
    return cutout_aug(images[i],images2[i],labels[i])
def RandomErasing(images,images2,labels):
    Re = re_aug.RandomErasing(probability = 1, sh = 0.4, r1 = 0.3, )
    
    B,_,_,_ = images.shape
    aug1 = []
    aug2 = []
    aug3 = []
    # spath = "/home/wrf/4TDisk/CD/SG/results/retrain/ADCD/Data_Aug_Op/"
    for i in range(B):
        
        img1_aug,img2_aug,label_aug = Re(images[i],images2[i],labels[i])
        aug1.append(img1_aug)
        aug2.append(img2_aug)
        aug3.append(label_aug)
        # break

    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return cg1, cg2, cg3
def RandomErasing2(images,images2,labels,i):
    Re = re_aug.RandomErasing(probability = 1, sh = 0.4, r1 = 0.3, )
    return Re(images[i],images2[i],labels[i])
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def CutMix(img1,img2,labels):
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    
    B,_,_,_ = img1.shape
    # for i in range(B):
    aug = np.random.choice([True, False])
    
    # bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)#随机产生一个box的四个坐标

    for i in range(B):
        lam = np.random.beta(1, 1)  #随机的lam
        bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)#随机产生一个box的四个坐标
        index = random.randint(0,B-1)
        num = np.random.choice(10)
        # if num<7:
            # image1 = img1[]
        img1[i, :, bbx1:bbx2, bby1:bby2] = img1[index, :, bbx1:bbx2, bby1:bby2]#将样本i中box的像素值填充该批数据

        img2[i, :, bbx1:bbx2, bby1:bby2] = img2[index, :, bbx1:bbx2, bby1:bby2]#将样本i中box的像素值填充该批数据

        labels[i, :, bbx1:bbx2, bby1:bby2] = labels[index, :, bbx1:bbx2, bby1:bby2]
        aug1.append(img1[i])
        aug2.append(img2[i])
        aug3.append(labels[i])

        # else:
        #     img1[i] , img2[i] ,labels[i] = img1[i] , img2[i] ,labels[i]
        #     aug1.append(img1[i])
        #     aug2.append(img2[i])
        #     aug3.append(labels[i])
        
    # return img1, img2, labels
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return cg1, cg2, cg3

def CutMix2(img1,img2,labels,i):
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    
    B,_,_,_ = img1.shape
    # for i in range(B):
    aug = np.random.choice([True, False])
    
    # bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)#随机产生一个box的四个坐标

    # for i in range(B):
    lam = np.random.beta(1, 1)  #随机的lam
    bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)#随机产生一个box的四个坐标
    index = random.randint(0,B-1)
    num = np.random.choice(10)
    # if num<7:
        # image1 = img1[]
    img1[i, :, bbx1:bbx2, bby1:bby2] = img1[index, :, bbx1:bbx2, bby1:bby2]#将样本i中box的像素值填充该批数据

    img2[i, :, bbx1:bbx2, bby1:bby2] = img2[index, :, bbx1:bbx2, bby1:bby2]#将样本i中box的像素值填充该批数据

    labels[i, :, bbx1:bbx2, bby1:bby2] = labels[index, :, bbx1:bbx2, bby1:bby2]
    # aug1.append(img1[i])
    # aug2.append(img2[i])
    # aug3.append(labels[i])

    #     # else:
    #     #     img1[i] , img2[i] ,labels[i] = img1[i] , img2[i] ,labels[i]
    #     #     aug1.append(img1[i])
    #     #     aug2.append(img2[i])
    #     #     aug3.append(labels[i])
        
    # # return img1, img2, labels
    # cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return img1[i], img2[i], labels[i]

def noisy(image1,image2):
    image_aug = image1.copy()
    image_aug2 = image2.copy()
    origin = img_as_float(image_aug)
    origin2 = img_as_float(image_aug2)
    # origin = Image.fromarray(np.uint8(img1_tensor.numpy().transpose(1,2,0)))
    # origin2 = Image.fromarray(np.uint8(img2_tensor.numpy().transpose(1,2,0)))
    image_aug =  util.random_noise(origin, mode="gaussian")
    image_aug2 =  util.random_noise(origin2, mode="gaussian")
    image_aug = Image.fromarray(np.uint8(image_aug*255))
    image_aug2 = Image.fromarray(np.uint8(image_aug2*255))
    return image_aug,image_aug2
def aug_v1(img1,img2,labels):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    preprocess2 = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations2
    ws = np.float32(
        np.random.dirichlet([1] * 3))
    B,_,_,_ = img1.shape
    # print(B)
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    # spath = "/home/wrf/4TDisk/CD/SG/results/retrain/ADCD/Data_Aug_Op/"
    for i in range(B):
        aug = np.random.choice([True, False])
        index = random.randint(0,B-1)
        # print('img1-shape:{}'.format(img1.shape))
        # if aug==False:
        #     mixed , mixed2 , label = img1[i] , img2[i] , labels[i]
        image1_ori = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
        image2_ori = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))    
        # image1_ori = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
        image1_ori = preprocess2(image1_ori)
        image2_ori = preprocess2(image2_ori)
        label_ori = labels[i]
        # print(label_ori.shape)
        image1 = img1[index]*labels[index] + (1-labels[index])*img1[i]
        # image2 = img2[i]
        image2 = img2[index]*labels[index] + (1-labels[index])*img2[i]
        label = labels[i]*(1-labels[index]) + labels[index]
        # print(label.shape)
        image1_c = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
        image2_c = Image.fromarray(np.uint8((image2*255).numpy().transpose(1,2,0)))
        img1_tensor = preprocess(image1_c)
        img2_tensor = preprocess(image2_c)
        # torchvision.utils.save_image(img1_tensor,os.path.join(spath, "image", "{}cp1.png".format(i)))
        # torchvision.utils.save_image(img1_tensor,os.path.join(spath, "label", "{}cp1.png".format(name[index])))
        # torchvision.utils.save_image(img2_tensor,os.path.join(spath, "image2", "{}cp2.png".format(i)))
        mixed = torch.zeros_like(img1_tensor)
        mixed2 = torch.zeros_like(img2_tensor)
        
        num = np.random.choice(10)
        if aug:
            aug1.append(image1_ori)
            aug2.append(image2_ori)
            aug3.append(label_ori)
            # print(' ')
        else:
            for j in range(3):
                idx = np.random.choice(5)
                # print(idx)
                if(idx==0):
                    
                    # image_aug , image_aug2 = AugMix2(image1,image2)
                    image_aug , image_aug2 = noisy(image1_c,image2_c)
                    # torchvision.utils.save_image(preprocess(image_aug),os.path.join(spath, "image", "{}{}{}op1.png".format(i,j,ws[j])))
                    # torchvision.utils.save_image(preprocess(image_aug2),os.path.join(spath, "image2", "{}{}{}op2.png".format(i,j,ws[j])))
                    mixed += ws[j] * preprocess2(image_aug)
                    mixed2 += ws[j] * preprocess2(image_aug2)
                    # aug1.append(mixed)
                    # aug2.append(mixed2)
                # elif idx==0 and (j==1 or j==2) :
                else :
                    # image_aug = op(image_aug, 1)
                    image_aug = image1_c.copy()
                    image_aug2 = image2_c.copy()
                    op = np.random.choice(aug_list)
                    image_aug = preprocess2(op(image_aug, 1))
                    image_aug2 = preprocess2(op(image_aug2, 1))
                    mixed += ws[j] * image_aug
                    mixed2 += ws[j] * image_aug2
            mixed = (1 - m) * image1 + m * mixed
            mixed2 = (1 - m) * image2 + m * mixed2
            # torchvision.utils.save_image(mixed,os.path.join(spath, "image", "{}afteraug1.png".format(m)))
            # torchvision.utils.save_image(mixed2,os.path.join(spath, "image", "{}afteraug2.png".format(m)))
            aug1.append(mixed)
            aug2.append(mixed2)
            aug3.append(label)
        # break

    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return cg1, cg2, cg3
def AugMix(img1,img2,labels,i):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations
    ws = np.float32(
        np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    B,_,_,_ = img1.shape
    # for i in range(B):
    # print(img1.shape)
    image1 = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
    image2 = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))
    img1_tensor = preprocess(image1)
    img2_tensor = preprocess(image2)
    mix = torch.zeros_like(img1_tensor)
    mix2 = torch.zeros_like(img2_tensor)

    aug = np.random.choice([True, False])
    index = random.randint(0,B-1)
    # if aug :
    num = np.random.choice(10)
    # if num>6:
    for k in range(3): # three paths
        image_aug = image1.copy()
        image_aug2 = image2.copy()
        depth = np.random.randint(1, 4)
        for _ in range(depth):

            op = np.random.choice(aug_list)
            image_aug = op(image_aug, 1)
            image_aug2 = op(image_aug2, 1)


        # Preprocessing commutes since all coefficients are convex

        mix += ws[k] * preprocess(image_aug)
        mix2 += ws[k] * preprocess(image_aug2)
    mixed = (1 - m) * img1_tensor + m * mix
    mixed2 = (1 - m) * img2_tensor + m * mix2
        # aug1.append(mixed)
        # aug2.append(mixed2)
        # aug3.append(labels[i])
        # else :
        #     aug1.append(img1[i])
        #     aug2.append(img2[i])
        #     aug3.append(labels[i])
    # cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image

    return mixed, mixed2, labels[i]
    # return mixed,mixed2
def AugMix2(img1,img2,labels):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    preprocess2 = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations
    ws = np.float32(
        np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    B,_,_,_ = img1.shape
    for i in range(B):

        image1 = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
        image2 = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))
        # print(img1[i].max(),np.max(image1))
        # Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0))).save('i1/{}'.format(name[i]))
        img1_tensor = preprocess(image1)
        img2_tensor = preprocess(image2)
        mix = torch.zeros_like(img1_tensor)
        mix2 = torch.zeros_like(img2_tensor)
        for k in range(3): # three paths
            image_aug = image1.copy()
            image_aug2 = image2.copy()
            depth = np.random.randint(1, 4)
            for _ in range(depth):

                op = np.random.choice(aug_list)
                image_aug = op(image_aug, 1)
                image_aug2 = op(image_aug2, 1)


            # Preprocessing commutes since all coefficients are convex

            mix += ws[k] * preprocess(image_aug)
            mix2 += ws[k] * preprocess(image_aug2)
        mixed = (1 - m) * img1_tensor + m * mix
        mixed2 = (1 - m) * img2_tensor + m * mix2
        aug = np.random.choice([True, False])
        # if aug:
        aug1.append(mixed)
        aug2.append(mixed2)
        aug3.append(labels[i])
        # else:
        #     aug1.append(mixed2)
        #     aug2.append(mixed)
        #     aug3.append(labels[i])
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image

    return cg1, cg2, cg3

def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)

def convaug(img1,img2,labels,i,idx):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations_all
 
    image1 = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
    image2 = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))
    # print(labels[i].shape)
    label = Image.fromarray(np.uint8((labels[i][0]*255).numpy()))
    image_aug = image1.copy()
    image_aug2 = image2.copy()
    label_aug = label.copy()
    if idx =='autocontrast':
        index = 0
    elif idx == 'shear_x':
        index = 5
    elif idx == 'shear_y':
        index = 6
    elif idx == 'translate_x' or idx == 'translate_y':
        index = 7
    elif idx == 'equalize':
        index = 1
    elif idx == 'posterize':
        index = 2
    elif idx =='solarize':
        index = 4
    op = aug_list[index]
    if index == 3:
        degrees = int_parameter(sample_level(3), 30)
        image_aug = op(image_aug, degrees)
        image_aug2 = op(image_aug2, degrees)
        label_aug = op(label_aug,degrees)
    elif index == 5:
        degrees = float_parameter(sample_level(3), 0.3)
        image_aug = op(image_aug, degrees)
        image_aug2 = op(image_aug2, degrees)
        label_aug = op(label_aug,degrees)
    elif index == 6:
        degrees = float_parameter(sample_level(3), 0.3)
        image_aug = op(image_aug, degrees)
        image_aug2 = op(image_aug2, degrees)
        label_aug = op(label_aug,degrees)
    elif index == 7:
        degrees = int_parameter(sample_level(3), 256 / 3)
        image_aug = op(image_aug, degrees)
        image_aug2 = op(image_aug2, degrees)
        label_aug = op(label_aug,degrees)
    elif index == 8:
        degrees = int_parameter(sample_level(3), 256 / 3)
        image_aug = op(image_aug, degrees)
        image_aug2 = op(image_aug2, degrees)
        label_aug = op(label_aug,degrees)
    # Preprocessing commutes since all coefficients are convex
    else:
        image_aug = op(image_aug, 3)
        image_aug2 = op(image_aug2, 3)
        label_aug = label
    return preprocess(image_aug),preprocess(image_aug2),preprocess(label_aug)
  


def MixUp(img1,img2,labels):
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    B,_,_,_ = img1.shape
    for i in range(B):
        aug = np.random.choice([True, False])
        index = random.randint(0,B-1)
        num = np.random.choice(10)
        # if num<7:
        #     aug1.append(img1[i])
        #     aug2.append(img2[i])
        #     aug3.append(labels[i])
        #     # image1 = img1[index]*m + (1-m)*img1[i]
        #     # # image2 = img2[i]
        #     # image2 = img2[index]*m + (1-m)*img2[i]
        #     # label = labels[i]
        #     # aug1.append(image1)
        #     # aug2.append(image2)
        #     # aug3.append(label)
        #     # print(' ')
        # else:
        image1 = img1[index]*(1-m) + m*img1[i]
        # image2 = img2[i]
        image2 = img2[index]*(1-m) + m*img2[i]
        label = labels[i]
        aug1.append(image1)
        aug2.append(image2)
        aug3.append(label) 
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    return cg1, cg2, cg3
def aug_v1d(img1,img2,labels):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations2
    ws = np.float32(
        np.random.dirichlet([1] * 3))
    B,_,_,_ = img1.shape
    # print(B)
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    # spath = "/home/wrf/4TDisk/CD/SG/results/retrain/ADCD/Data_Aug_Op/"
    for i in range(B):
        aug = np.random.choice([True, False])
        index = random.randint(0,B-1)
        image1_ori = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
        image2_ori = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))    
        # image1_ori = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
        image1_ori = preprocess(image1_ori)
        image2_ori = preprocess(image2_ori)
        label_ori = labels[i]
        # print(label_ori.shape)
        image1 = img1[index]*labels[index] + (1-labels[index])*img1[i]
        # image2 = img2[i]
        image2 = img2[index]*labels[index] + (1-labels[index])*img2[i]
        label = labels[i]*(1-labels[index]) + labels[index]
        # print(label.shape)
        image1_c = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
        image2_c = Image.fromarray(np.uint8((image2*255).numpy().transpose(1,2,0)))
        img1_tensor = preprocess(image1_c)
        img2_tensor = preprocess(image2_c)
        mixed = torch.zeros_like(img1_tensor)
        mixed2 = torch.zeros_like(img2_tensor)
        
        num = np.random.choice(10)
        if num<7:
            aug1.append(image1_ori)
            aug2.append(image2_ori)
            aug3.append(label_ori)
            # print(' ')
        else:
            for j in range(3):
                idx = np.random.choice(5)
                depth = np.random.randint(1, 4)
                for _ in range(depth):
                # print(idx)
                    if(idx==0):
                        
                        # image_aug , image_aug2 = AugMix2(image1,image2)
                        image_aug , image_aug2 = noisy(image1_c,image2_c)
                        image_aug , image_aug2 = preprocess(image_aug),preprocess(image_aug2)
                        # mixed += ws[j] * image_aug
                    else :
                        # image_aug = op(image_aug, 1)
                        image_aug = image1_c.copy()
                        image_aug2 = image2_c.copy()
                        op = np.random.choice(aug_list)
                        image_aug = preprocess(op(image_aug, 1))
                        image_aug2 = preprocess(op(image_aug2, 1))
                        # print(image_aug)
                mixed += ws[j] * image_aug
                mixed2 += ws[j] * image_aug2
                  
            mixed = (1 - m) * image1 + m * mixed
            mixed2 = (1 - m) * image2 + m * mixed2
            aug1.append(mixed)
            aug2.append(mixed2)
            aug3.append(label)
        # break

    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

    return cg1, cg2, cg3
def cutmix(view1, view2,bbx1, bby1, bbx2, bby2):
    lam = np.random.uniform(low=0.0, high=1.0)
    lam2 = np.random.uniform(low=0.0, high=1.0)

    _, h, w = view1.shape
 
    view1[:, bbx1:bbx2, bby1:bby2] = view2[:, bbx1:bbx2, bby1:bby2]
    return view1

def random_bbox(lam, H, W):
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def cmixup(img1,img2,m):
        # m = np.float32(np.random.beta(1, 1))
    aug_img1 = m*img1+(1-m)*img2
    return aug_img1
def custom_crop(img1, img2,label, size, scale_range, ratio_range):
    # 定义随机数生成器
    seed = random.randint(0, 2**32)
    random_gen = random.Random(seed)

    # 随机生成裁剪参数
    aspect_ratio = random_gen.uniform(*ratio_range)
    area = img1.size[0] * img1.size[1]
    for _ in range(10):
        target_area = area * random_gen.uniform(*scale_range)
        w = int(round((target_area * aspect_ratio) ** 0.5))
        h = int(round((target_area / aspect_ratio) ** 0.5))
        if 0 < w <= img1.size[0] and 0 < h <= img1.size[1]:
            x1 = random_gen.randint(0, img1.size[0] - w)
            y1 = random_gen.randint(0, img1.size[1] - h)
            x2 = x1 + w
            y2 = y1 + h
            break
    else:
        # 如果无法得到合适的裁剪参数，则使用中心裁剪
        x1 = (img1.size[0] - size) // 2
        y1 = (img1.size[1] - size) // 2
        x2 = x1 + size
        y2 = y1 + size

    # 对一对图像进行裁剪，保证两张图像使用相同的随机数生成器
    crop1 = img1.crop((x1, y1, x2, y2))
    crop2 = img2.crop((x1, y1, x2, y2))
    crop3 = label.crop((x1, y1, x2, y2))
    return crop1, crop2,crop3


def CropMix(images1,images2,labels):
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    transform2 = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    aug1 = []
    aug2 = []
    aug3 = []
    B, _, _, _ = images1.shape
    for j in range(B):
        img1, img2, label = images1[j],images2[j],labels[j]
        image1_ori = Image.fromarray(np.uint8((img1*255).numpy().transpose(1,2,0)))
        image2_ori = Image.fromarray(np.uint8((img2*255).numpy().transpose(1,2,0)))    
        # image1_ori = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
        image1_ori = transform(image1_ori)
        image2_ori = transform(image2_ori)
        # print(img1.max())
        num = np.random.choice(10)
        if num<7:
            aug1.append(image1_ori)
            aug2.append(image2_ori)
            aug3.append(label)
        else:
            img1, img2, label = transforms.ToPILImage()(img1.float()),transforms.ToPILImage()(img2.float()),transforms.ToPILImage()(label.float())
            for i in range(3):
                crop_size = 256
                if i == 0:
                    scale_range = (0.01, 0.34)
                elif i == 1:
                    scale_range = (0.34, 0.67)
                elif i == 2:
                    scale_range = (0.67, 1)
                ratio_range = (3.0/4, 4.0/3)
                transformed_img1, transformed_img2 ,transformed_label= custom_crop(img1, img2,label, crop_size, scale_range, ratio_range)
                new_img1 = transform(transformed_img1)
                # torchvision.utils.save_image(new_img1,'a.png')
                new_img2 = transform(transformed_img2)
                # torchvision.utils.save_image(new_img2,'b.png')
                new_label = transform2(transformed_label)
                # print(label.max())
                _,new_label = cv2.threshold(np.uint8((new_label*255).numpy()), 128, 1, cv2.THRESH_BINARY)
                # new_label = (new_label*255).astype(np.uint8)
                new_label = Image.fromarray(new_label[0]*255)
                # print(new_label.max())
                new_label = transforms.ToTensor()(new_label)

                if i == 0:
                    new_images11,new_images21,label1 = new_img1,new_img2,new_label
                    # torchvision.utils.save_image(new_images11,'a1.png')
                    # torchvision.utils.save_image(new_images21,'b1.png')
                    # torchvision.utils.save_image(label1,'c1.png')
                elif i == 1:
                    new_images12,new_images22,label2 = new_img1,new_img2,new_label
 #                   lam = np.random.uniform(low=0.0, high=1.0)
#                    bbx1, bby1, bbx2, bby2 = random_bbox(lam,256,256)
                    m = np.float32(np.random.beta(1, 1))
                    mixed1 , mixed2, mixed3 = cmixup(new_images11,new_images12,m),cmixup(new_images21,new_images22,m),label2
                    # torchvision.utils.save_image(mixed1,'a2.png')
                    # torchvision.utils.save_image(mixed2,'b2.png')
                    # torchvision.utils.save_image(mixed3,'c2.png')
                elif i == 2:
                    new_images13,new_images23,label3 = new_img1,new_img2,new_label
 #                   lam = np.random.uniform(low=0.0, high=1.0)
#                    bbx1, bby1, bbx2, bby2 = random_bbox(lam,256,256)
  #                  mixed1 , mixed2, mixed3 = cutmix(mixed1,new_images13,bbx1, bby1, bbx2, bby2),cutmix(mixed2,new_images23,bbx1, bby1, bbx2, bby2),cutmix(mixed3,label3,bbx1, bby1, bbx2, bby2)
                    m = np.float32(np.random.beta(1, 1))
                    mixed1 , mixed2, mixed3 = cmixup(mixed1,new_images13,m),cmixup(mixed2,new_images23,m),label3
                    aug1.append(mixed1)
                    aug2.append(mixed2)
                    aug3.append(mixed3.byte())
        # torchvision.utils.save_image(mixed1,'a3.png')
        # torchvision.utils.save_image(mixed2,'b3.png')
        # torchvision.utils.save_image(mixed3,'c3.png')
    # print(new_label.max())
    cg1 , cg2 , cg3 = torch.stack(aug1),torch.stack(aug2),torch.stack(aug3)
    return cg1 , cg2 , cg3
def MUM(images,images2,labels,i):
    h,w,c=256,256,3
    B,_,_,_=images.shape 
    bs = B
    ng = 4
    nt = 4
    aug1 = []
    aug2 = []
    aug3 = []
    # for i in range(B):
    num = np.random.choice(10)
    # if num==111111111111:
    #     return images,images2,labels
    # else:
    mask = torch.argsort(torch.rand(bs // ng, ng, nt, nt), dim=1)
    img_mask = mask.view(bs // ng, ng, 1, nt, nt)
    img_maskl = mask.view(bs // ng, ng, 1, nt, nt)
    img_mask = img_mask.repeat_interleave(3, dim=2)
    img_maskl = img_maskl.repeat_interleave(1, dim=2)
    img_mask = img_mask.repeat_interleave(h // nt, dim=3)
    img_maskl = img_maskl.repeat_interleave(h // nt, dim=3)
    img_mask = img_mask.repeat_interleave(w // nt, dim=4)
    img_maskl = img_maskl.repeat_interleave(w // nt, dim=4)

    img_tiled = images.view(bs // ng, ng, c, h, w)
    img_tiled = torch.gather(img_tiled, dim=1, index=img_mask)
    img_tiled = img_tiled.view(bs, c, h, w)

    img_tiled2 = images2.view(bs // ng, ng, c, h, w)
    img_tiled2 = torch.gather(img_tiled2, dim=1, index=img_mask)
    img_tiled2 = img_tiled2.view(bs, c, h, w)

    img_tiledl = labels.view(bs // ng, ng, 1, h, w)
    # print(img_tiledl.shape)
    img_tiledl = torch.gather(img_tiledl,  dim=1,index=img_maskl)
    img_tiledl = img_tiledl.view(bs, 1, h, w)
    # aug1.append(img_tiled)
    # aug2.append(img_tiled2)
    # aug3.append(img_tiledl)
    # cg1 , cg2 , cg3 = torch.stack(aug1),torch.stack(aug2),torch.stack(aug3)
    # return img_tiled , img_tiled2 , img_tiledl
    return img_tiled[i] , img_tiled2[i] , img_tiledl[i]
# def find_ith_smallest_region(image, x, y, i):
#     h, w,_ = image.shape
#     regions = []
#     for r in range(0,h-y+1,30):
#         for c in range(0,w-x+1,30):
#             region_sum = np.sum(image[r:r+y, c:c+x])
#             regions.append(((r,c), region_sum))
#     regions.sort(key=lambda x: x[1])
#     return regions[i-1][0]
def find_change_labels(labels):
    change_indices = []
    b,_,_,_ = labels.shape
    for index in range(b):
        max_label = torch.sum(labels[index])
        if max_label != 0:
            change_indices.append(index)
    return change_indices
def find_ith_smallest_region(image, x, y, i):
    h, w= image.shape
    regions = []
    for r in range(h-y+1,-1,-20):
        for c in range(w-x+1,-1,-20):
            region_sum = np.sum(image[r:r+y, c:c+x])
            if region_sum==0:
                # return (r,c)
                regions.append(((r,c), region_sum))
    
    # print(selected_data[0])
    # regions.sort(key=lambda x: x[1],reverse=True)
    # print(regions)
    # print('--------')
    # print( regions[len(regions)-1][0])
    # return selected_data[0]
    if regions:
        selected_data = random.choice(regions)
        return selected_data[0]
    else:
        return None

def MGM(img1,img2,labels,name,pre_dict):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    # mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    # std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    preprocess2 = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        # transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
        ])
    B,_,_,_ = img1.shape
    aug1 = []
    aug2 = []
    aug3 = []
    # aug_list = augmentations.augmentations_all
    # print(labels.shape)
    change_img = find_change_labels(labels)
    for i in range(B):
        aug = np.random.choice([True, False])
        img1_ori = Image.fromarray(np.uint8((img1[i]*255).numpy()).transpose(1,2,0) )
        img2_ori = Image.fromarray(np.uint8((img2[i]*255).numpy()).transpose(1,2,0) )
        # if epoch>1:
        if len(change_img)!=0:
            index = random.choice(change_img)
            re = 0
            lab = np.uint8((labels[i][0]*255).numpy())
            lab2 = np.uint8((labels[index][0]*255).numpy())
            image = np.uint8((img1[i]*255).numpy())
            image_aug = np.uint8((img1[index]*255).numpy())
            image2 = np.uint8((img2[i]*255).numpy())
            image_aug2 = np.uint8((img2[index]*255).numpy())
            test = np.zeros_like(lab)
            image_a = image.transpose(1,2,0) 
            image2_a = image2.transpose(1,2,0)
            # image_a = image_aug.transpose(1,2,0) 
            # image2_a = image_aug2.transpose(1,2,0)
            
            contours, _ = cv2.findContours(lab2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = 0
            # temp_xor = cv2.imread('/home/dl/1Tdriver/wjd/ChangeFormer-main/or_result2/{}'.format(name[i]), cv2.COLOR_BGR2GRAY)
            temp_xor = pre_dict[name[i]]
            # if temp_xor.max!=0:
            #     # print(diff.max())
            #         cv2.imwrite('image_aug.png',temp_xor)
      
            for contour_idx in range(len(contours)):
                x1, y1, w, h = cv2.boundingRect(contours[contour_idx])
                # print(w,h)
                crop_label = lab2[y1:y1+h, x1:x1+w]
                window_size = crop_label.shape
                count = count + 1
                if np.sum(crop_label)<50000:
                    continue
                num = 0
                re +=1
        
                diff = cv2.bitwise_or(temp_xor,lab)
                # if diff.max!=0:
                # # print(diff.max())
                #     cv2.imwrite('image_aug.png',diff)
                min_region = find_ith_smallest_region(diff,window_size[0],window_size[1],1)
                # diff_to_window_label = diff[min_region[0]:min_region[0]+window_size[1], min_region[1]:min_region[1]+window_size[0]]
                if min_region!=None:
                    num = num + 1
                    offset_x = min_region[1] - x1
                    offset_y = min_region[0] - y1
                    #变化矩阵
                    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    temp = np.zeros((256, 256, 3), dtype='uint8')
                    cv2.drawContours(temp, contours, contour_idx, (255, 255, 255), -1)
                    temp = temp//255
                    temp_img1 = temp*(image_aug.transpose(1,2,0))
                    temp_img2 = temp*(image_aug2.transpose(1,2,0))
                    shifted1 = cv2.warpAffine(temp_img1, M, (256, 256))
                    shifted2 = cv2.warpAffine(temp_img2, M, (256, 256))
                    shifted_label = cv2.warpAffine(temp, M, (256, 256))
                    cv2.drawContours(test, contours, contour_idx, (255, 255, 255), -1,offset=[offset_x,offset_y])
                    lab = cv2.bitwise_or(test,lab)
                    # aug = np.random.choice([True, False])
                    if aug:
                        image_a = shifted_label*shifted1 + (1-shifted_label)*image_a
                        image2_a = shifted_label*shifted2 + (1-shifted_label)*image2_a
                    else:
                        image_a = shifted_label*shifted2 + (1-shifted_label)*image_a
                        image2_a = shifted_label*shifted1 + (1-shifted_label)*image2_a
                    # cv2.imwrite('image_aug.png',image_a*255/2.64)
            # print(image_a.shape)
            # Image.fromarray(image_a).save('a/{}'.format(name[i]))
            # Image.fromarray(image2_a).save('b/{}'.format(name[i]))
            # Image.fromarray(lab).save('l/{}'.format(name[i]))
            image_a = preprocess2(Image.fromarray(image_a))
            # print('------------{}'.format(image_a.shape))
            # Image.fromarray(np.uint8(image_a.numpy()).transpose(1,2,0) ).save('a/{}'.format(name[i]))
            image2_a = preprocess2(Image.fromarray(image2_a))
            augp = np.random.choice([True, False])
            if augp:
                aug1.append(image_a)
                aug2.append(image2_a)
                aug3.append( preprocess(Image.fromarray(lab)))
            else:
                aug1.append(image2_a)
                aug2.append(image_a)
                aug3.append( preprocess(Image.fromarray(lab)))
       
        else :

            # aug1.append(transforms.Normalize([.5, .5, .5], [.5, .5, .5])(img2[i]))
            # aug2.append(transforms.Normalize([.5, .5, .5], [.5, .5, .5])(img1[i]))
            aug1.append((img2[i]))
            aug2.append((img1[i]))
            aug3.append(labels[i])
     
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image

    return cg1, cg2, cg3

def MGM2(img1,img2,labels,i,name,action,pre_dict):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    # mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    # std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    preprocess2 = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        # transforms.Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229])
        ])
    # B,_,_,_ = img1.shape
    aug1 = []
    aug2 = []
    aug3 = []
    # aug_list = augmentations.augmentations_all
    # print(labels.shape)
    change_img = find_change_labels(labels)
    # for i in range(B):
    aug = np.random.choice([True, False])
    img1_ori = Image.fromarray(np.uint8((img1[i]*255).numpy()).transpose(1,2,0) )
    img2_ori = Image.fromarray(np.uint8((img2[i]*255).numpy()).transpose(1,2,0) )
    # if epoch>1:
    if len(change_img)!=0:
        index = random.choice(change_img)
        re = 0
        lab = np.uint8((labels[i][0]*255).numpy())
        lab2 = np.uint8((labels[index][0]*255).numpy())
        image = np.uint8((img1[i]*255).numpy())
        image_aug = np.uint8((img1[index]*255).numpy())
        image2 = np.uint8((img2[i]*255).numpy())
        image_aug2 = np.uint8((img2[index]*255).numpy())
        test = np.zeros_like(lab)
        image_a = image.transpose(1,2,0) 
        image2_a = image2.transpose(1,2,0)
        # image_a = image_aug.transpose(1,2,0) 
        # image2_a = image_aug2.transpose(1,2,0)
        
        contours, _ = cv2.findContours(lab2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        # print(pre_dict)
        temp_xor = pre_dict[name[i]]
        # print(temp_xor.max())
        for contour_idx in range(len(contours)):
            x1, y1, w, h = cv2.boundingRect(contours[contour_idx])
            # print(w,h)
            crop_label = lab2[y1:y1+h, x1:x1+w]
            window_size = crop_label.shape
            count = count + 1
            if np.sum(crop_label)<50000:
                continue
            num = 0
            re +=1
            # print(temp_xor.shape,lab.shape)
            diff = cv2.bitwise_or(temp_xor,lab)
            # print(diff)
            # if diff.max!=0:
            #     # print(diff.max())
            #     cv2.imwrite('image_aug.png',diff)
            min_region = find_ith_smallest_region(diff,window_size[0],window_size[1],1)
            # diff_to_window_label = diff[min_region[0]:min_region[0]+window_size[1], min_region[1]:min_region[1]+window_size[0]]
            if min_region!=None:
                num = num + 1
                offset_x = min_region[1] - x1
                offset_y = min_region[0] - y1
                #变化矩阵
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                temp = np.zeros((256, 256, 3), dtype='uint8')
                cv2.drawContours(temp, contours, contour_idx, (255, 255, 255), -1)
                temp = temp//255
                temp_img1 = temp*(image_aug.transpose(1,2,0))
                temp_img2 = temp*(image_aug2.transpose(1,2,0))
                shifted1 = cv2.warpAffine(temp_img1, M, (256, 256))
                shifted2 = cv2.warpAffine(temp_img2, M, (256, 256))
                shifted_label = cv2.warpAffine(temp, M, (256, 256))
                cv2.drawContours(test, contours, contour_idx, (255, 255, 255), -1,offset=[offset_x,offset_y])
                lab = cv2.bitwise_or(test,lab)
                # aug = np.random.choice([True, False])
                if aug:
                    image_a = shifted_label*shifted1 + (1-shifted_label)*image_a
                    image2_a = shifted_label*shifted2 + (1-shifted_label)*image2_a
                else:
                    image_a = shifted_label*shifted2 + (1-shifted_label)*image_a
                    image2_a = shifted_label*shifted1 + (1-shifted_label)*image2_a
  
        image_a = preprocess2(Image.fromarray(image_a))
        image2_a = preprocess2(Image.fromarray(image2_a))
        # print('-aaaaaaaaaaa')
        # if action==0:
            # return image_a,image2_a,preprocess(Image.fromarray(lab))
        # elif action==1:
            # return image2_a,image_a,preprocess(Image.fromarray(lab))
        # print('-------------')
        augp = np.random.choice([True, False])
        if augp:
            return image_a,image2_a,preprocess(Image.fromarray(lab))
        else:
            return image2_a,image_a,preprocess(Image.fromarray(lab))
    else :
        return img2[i], img1[i], labels[i]

def MaskMix(img1,img2,labels):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    preprocess2 = transforms.Compose([transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    aug_list = augmentations.augmentations2
    ws = np.float32(
        np.random.dirichlet([1] * 3))
    B,_,_,_ = img1.shape
    # print(B)
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []
    aug3 = []
    change_img = find_change_labels(labels)
    if len(change_img)!=0:
    # spath = "/home/wrf/4TDisk/CD/SG/results/retrain/ADCD/Data_Aug_Op/"
        for i in range(B):
            # change_img = find_change_labels(labels)
            index = random.choice(change_img)
            aug = np.random.choice([True, False])
            # index = random.randint(0,B-1)
            
            if random.random() < 0.3:
                image1_ori = Image.fromarray(np.uint8((img1[i]*255).numpy().transpose(1,2,0)))
                image2_ori = Image.fromarray(np.uint8((img2[i]*255).numpy().transpose(1,2,0)))    
                # image1_ori = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
                image1_ori = preprocess2(image1_ori)
                image2_ori = preprocess2(image2_ori)
                label_ori = labels[i]
                # print(label_ori.shape)
                image1 = img1[index]*labels[index] + (1-labels[index])*img1[i]
                # image2 = img2[i]
                image2 = img2[index]*labels[index] + (1-labels[index])*img2[i]
                label = labels[i]*(1-labels[index]) + labels[index]
                # print(label.shape)
                image1_c = Image.fromarray(np.uint8((image1*255).numpy().transpose(1,2,0)))
                image2_c = Image.fromarray(np.uint8((image2*255).numpy().transpose(1,2,0)))
                img1_tensor = preprocess(image1_c)
                img2_tensor = preprocess(image2_c)
                mixed = torch.zeros_like(img1_tensor)
                mixed2 = torch.zeros_like(img2_tensor)
                
                for j in range(3):
                    idx = np.random.choice(5)
                    # print(idx)
                    if(idx==0):
                        image_aug , image_aug2 = noisy(image1_c,image2_c)
                        mixed += ws[j] * preprocess2(image_aug)
                        mixed2 += ws[j] * preprocess2(image_aug2)
                    else :
                        image_aug = image1_c.copy()
                        image_aug2 = image2_c.copy()
                        op = np.random.choice(aug_list)
                        image_aug = preprocess2(op(image_aug, 1))
                        image_aug2 = preprocess2(op(image_aug2, 1))
                        mixed += ws[j] * image_aug
                        mixed2 += ws[j] * image_aug2
                mixed = (1 - m) * image1 + m * mixed
                mixed2 = (1 - m) * image2 + m * mixed2
                aug1.append(mixed)
                aug2.append(mixed2)
                aug3.append(label)
            else:
                aug1.append(img1[i])
                aug2.append(img2[i])
                aug3.append(labels[i])
        cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    # ug1, ug2 = torch.stack(auf1).cuda(), torch.stack(auf2).cuda()

        return cg1, cg2, cg3
        # break
    else:

        return img1, img2, labels
def getbox_new(size, dxy):
    W = size[2]
    H = size[3]
    # print(W,H)
    cx = np.random.randint(0, W, size[0])
    cy = np.random.randint(0, H, size[0])

    bbx1 = np.clip(cx - dxy[0] // 2, 0, W)
    bbx2 = np.clip(bbx1 + dxy[0], 0, W)
    bbx1 = np.clip(bbx2 - dxy[0], 0, W)
    bby1 = np.clip(cy - dxy[1] // 2, 0, H)
    bby2 = np.clip(bby1 + dxy[1], 0, H)
    bby1 = np.clip(bby2 - dxy[1], 0, H)
    
    return bbx1, bby1, bbx2, bby2

from models.tokenmix import Mixup_Token
from models.tokenmix2 import Mixup_Token2
def TokenMix(img1,img2,labels):
    mixup_fn = Mixup_Token2(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=2, mask_type='block')
    B,C,W,H = labels.shape
    label = labels.reshape((1,B,C,W,H))
    img1_aug,img2_aug,label_aug,label = mixup_fn(img1,img2,labels.float(),label)
    return img1_aug,img2_aug,label_aug
def TokenMix2(img1,img2,labels,i):
    mixup_fn = Mixup_Token(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=2, mask_type='block')
    B,C,W,H = labels.shape
    label = labels.reshape((1,B,C,W,H))
    img1_aug,img2_aug,label_aug,label = mixup_fn(img1,img2,labels.float(),label)
    return img1_aug[i],img2_aug[i],label_aug[i]      
def GriMask_AUG2(img1,img2,labels,i):
    gridmask = Grid(96,224, 360, False,0.6,1,1)
    img1_aug,img2_aug,label_aug = gridmask(img1[i],img2[i],labels[i])
    return img1_aug,img2_aug,label_aug

def GriMask_AUG(img1,img2,labels):
    gridmask = Grid(96,224, 360, False,0.6,1,1)
    B,_,_,_ = img1.shape
    aug1 = []
    aug2 = []
    aug3 = []
    for i in range(B):

        img1_aug,img2_aug,label_aug = gridmask(img1[i],img2[i],labels[i])
        aug1.append(img1_aug)
        aug2.append(img2_aug)
        aug3.append(label_aug)
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image

    return cg1, cg2, cg3
    # return img1_aug,img2_aug,label_aug
def apply_augmentation(img1,img2,labels,i, action,name,pre_dict):
    # 在这里实现你的数据增强操作，这里仅作示例
    # action += 
    # if action == 0:
    #     return img1[i], img2[i], labels[i]  # 无操作
    # elif action == 1:
    #     return  DiffMaskMix2(img1,img2,labels,i,name,action) 
    # elif action == 2:
    #     return   DiffMaskMix2(img1,img2,labels,i,name,action)
    # ['Ours','AugMix','MUM','CutMix','TokenMix','MixUp','W/OAug','GridMask']
    if action == 0:
        return  MGM2(img1,img2,labels,i,name,action,pre_dict) 
    # elif action == 1:
        # return   DiffMaskMix2(img1,img2,labels,i,name,action)
    elif action == 1:
        return   AugMix(img1,img2,labels,i)
    # elif action == 3:
        # return   img1[i],img2[i],labels[i]
    elif action == 2:
        return   MUM(img1,img2,labels,i)
    elif action == 3:
        return CutMix2(img1,img2,labels,i)
    elif action == 4:
        return TokenMix2(img1,img2,labels,i)
    elif action == 5:
        m = np.float32(np.random.beta(1, 1))
        B,_,_,_ = img1.shape
        index = random.randint(0,B-1)
        img1_aug = img1[i]*m + img1[index]*(1-m)
        img2_aug = img2[i]*m + img2[index]*(1-m)
        return img1_aug,img2_aug,labels[i]
    elif action == 6:
        return img1[i],img2[i],labels[i]
    elif action == 7:
        return GriMask_AUG2(img1,img2,labels,i)
    elif action == 8:
        return CutOut2(img1,img2,labels,i)
    elif action == 9:
        return RandomErasing2(img1,img2,labels,i)
    elif action == 6:
        return convaug(img1,img2,labels,i,action-4)
    # elif action == 7:
    #     return convaug(img1,img2,labels,i,action-4)
    # elif action == 8:
    #     return convaug(img1,img2,labels,i,action-4)
    # elif action == 9:
    #     return convaug(img1,img2,labels,i,action-4)
    # elif action == 10:
    #     return convaug(img1,img2,labels,i,action-4)
    # elif action == 11:
    #     return convaug(img1,img2,labels,i,action-4)
    # if action == 13:
    #     return DiffMaskMix2(img1,img2,labels,i,name,action) 
    # else:
    #     return convaug(img1,img2,labels,i,action)

def apply_augmentation2(img1,img2,labels,i, action,name,pre_dict):
    # 在这里实现你的数据增强操作，这里仅作示例
    # action += 
    # if action == 0:
    #     return img1[i], img2[i], labels[i]  # 无操作
    # elif action == 1:
    #     return  DiffMaskMix2(img1,img2,labels,i,name,action) 
    # elif action == 2:
    #     return   DiffMaskMix2(img1,img2,labels,i,name,action)
    # ['Ours','AugMix','MUM','CutMix','TokenMix','MixUp','W/OAug','GridMask']
    if action == 'Ours':
        return  MGM2(img1,img2,labels,i,name,action,pre_dict) 
    # elif action == 1:
        # return   DiffMaskMix2(img1,img2,labels,i,name,action)
    elif action == 'AugMix':
        return   AugMix(img1,img2,labels,i)
    # elif action == 3:
        # return   img1[i],img2[i],labels[i]
    elif action == 'MUM':
        return   MUM(img1,img2,labels,i)
    elif action == "CutMix":
        return CutMix2(img1,img2,labels,i)
    elif action == "TokenMix":
        return TokenMix2(img1,img2,labels,i)
    elif action == "MixUp":
        m = np.float32(np.random.beta(1, 1))
        B,_,_,_ = img1.shape
        index = random.randint(0,B-1)
        img1_aug = img1[i]*m + img1[index]*(1-m)
        img2_aug = img2[i]*m + img2[index]*(1-m)
        return img1_aug,img2_aug,labels[i]
    elif action == "W/OAug":
        return img1[i],img2[i],labels[i]
    elif action == "GridMask":
        return GriMask_AUG2(img1,img2,labels,i)
    elif action == "ResizeMix":
        return ResizeMix_i(img1,img2,labels,i)
    elif action == "8":
        return CutOut2(img1,img2,labels,i)
    
def apply_augmentation3(img1,img2,labels,i, action,name,pre_dict):
    
    if action == 'Ours':
        return  MGM2(img1,img2,labels,i,name,action,pre_dict) 
    # elif action == 1:
        # return   DiffMaskMix2(img1,img2,labels,i,name,action)
    elif action == 'AugMix':
        return   AugMix(img1,img2,labels,i)
    # elif action == 3:
        # return   img1[i],img2[i],labels[i]
    elif action == 'MUM':
        return   MUM(img1,img2,labels,i)
    elif action == "MixUp":
        m = np.float32(np.random.beta(1, 1))
        B,_,_,_ = img1.shape
        index = random.randint(0,B-1)
        img1_aug = img1[i]*m + img1[index]*(1-m)
        img2_aug = img2[i]*m + img2[index]*(1-m)
        return img1_aug,img2_aug,labels[i]
    elif action == "W/OAug":
        return img1[i],img2[i],labels[i]
    elif action == "GridMask":
        return GriMask_AUG2(img1,img2,labels,i)
    elif action == "ResizeMix":
        return ResizeMix_i(img1,img2,labels,i)
    elif action == "8":
        return CutOut2(img1,img2,labels,i)
    elif action == "a":
        return RandomErasing2(img1,img2,labels,i)

def RL_Aug(img1,img2,labels,name,a2c_net):
    B,_,_,_ = img1.shape
    aug1 = []
    aug2 = []
    aug3 = []
    # resnet = resnet18()
    resnet = tmodels.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    img1_fea = resnet(img1)
    img2_fea = resnet(img2)
    change_fea = torch.abs(img1_fea - img2_fea)
    for i in range(B):
        state = change_fea[i]
        action_probs, _ = a2c_net(state)
        action = torch.multinomial(action_probs, 1).item()
        img1_aug, img2_aug, label_aug = apply_augmentation(img1, img2, labels, i, action, name)
        aug1.append(img1_aug)
        aug2.append(img2_aug)
        aug3.append(label_aug)
    cg1, cg2, cg3 = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
    return cg1, cg2, cg3
class CDTrainer():

    def __init__(self, args, dataloaders):

        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        # print(self.device)
        self.args = args
        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr
        self.a2c = ActorCriticV5(args.action_num).to(self.device)
        self.optimizer_a2c = optim.Adam(self.a2c.parameters(), lr=0.001)
        # define optimizers
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        self.max_reward = 0
        # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.9, 0.999))
        self.val = iter(utils.get_loader_rlval(self.args.data_name, img_size=self.args.img_size,
                                  batch_size=self.args.batch_size, is_train=False,
                                  split='trainvalc'))

        self.num_actions = args.action_num
        self.reduct_rates = args.reduce_rate
        self.data_paths = args.data_path
        self.warm_ups = args.warm_up
        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.G_pred1 = None
        self.G_pred2 = None
        self.G_pred3 = None

        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.history_loss = {}
        self.history_loss_test = {}
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # define the loss functions
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred1, self.G_pred2, self.G_middle1, self.G_middle2 = self.net_G(img_in1, img_in2) 
        self.G_pred = self.G_pred1 + self.G_pred2
       
    def _forward_pass2(self, batch,epoch):
        self.batch = batch
        # gt = self.batch['L'].to(self.device).long()
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        # self.G_pred = self.net_G(img_in1, img_in2)
        self.G_pred1, self.G_pred2, self.G_middle1, self.G_middle2 = self.net_G(img_in1, img_in2) 
        self.G_pred = self.G_pred1 + self.G_pred2
        # loss = self._pxl_loss(self.G_pred, self.batch['L'].to(self.device).long())
        # print(self.G_pred)
        # print(self.G_pred[0].shape)
        c_num,_,_,_ = self.G_pred.shape
        for i in range(c_num):
            # print(self.G_pred[i].max(0)[1].data.squeeze(0).byte(), batch['L'][i])
            # or_result = torch.logical_or(self.G_pred[i].max(0)[1].data.squeeze(0).byte(), batch['L'][i].cuda()).cpu().numpy()
            # print(self.G_pred[i].max(0)[1].data.squeeze(0).byte().shape)
            or_result = self.G_pred[i].max(0)[1].data.squeeze(0).byte().cpu().numpy()
            Image.fromarray(np.uint8(or_result * 255)).save(os.path.join('./or_result/{}'.format(batch['name'][i])))
    def _forward_pass3(self, batch , loss_dict,pre_dict):
        # self.batch = batch
        img_in1 = batch['A'].to(self.device)
        # print(img_in1.shape)
        img_in2 = batch['B'].to(self.device)
        b,_,_,_ = img_in1.shape
        G_pred1, G_pred2, G_middle1, G_middle2 = self.net_G(img_in1, img_in2) 
        G_pred = G_pred1 + G_pred2
       
        for i in range(b):
            or_result = G_pred[i].max(0)[1].data.squeeze(0).byte().cpu().numpy()
            pre_dict[batch['name'][i]] = or_result*255
            gt = batch['L'][i].to(self.device).float()
            G_loss =  self._pxl_loss(G_pred1[i].reshape(1,2,256,256), gt) + self._pxl_loss(G_pred2[i].reshape(1,2,256,256), gt) + 0.5*(self._pxl_loss(G_middle1[i].reshape(1,2,256,256), gt)+self._pxl_loss(G_middle2[i].reshape(1,2,256,256), gt))

            loss_dict[batch['name'][i]] = G_loss.detach().cpu() 
        
        # return loss_dict
    def _forward_pass_loss(self, batch ):
        # self.batch = batch
        img_in1 = batch['A'].to(self.device)
        # print(img_in1.shape)
        img_in2 = batch['B'].to(self.device)
        loss_list = []
        run_acc_list = []
        G_pred1, G_pred2, G_middle1, G_middle2 = self.net_G(img_in1, img_in2) 
        G_pred = G_pred1 + G_pred2
        
        B,_,_,_ = img_in2.shape
        for i in range(B):
           
            gt = batch['L'][i].to(self.device).float()
            G_loss =  self._pxl_loss(G_pred1[i].reshape(1,2,256,256), gt) + self._pxl_loss(G_pred2[i].reshape(1,2,256,256), gt) + 0.5*(self._pxl_loss(G_middle1[i].reshape(1,2,256,256), gt)+self._pxl_loss(G_middle2[i].reshape(1,2,256,256), gt))

            loss_list.append([G_loss.detach().cpu()])
            target = batch['L'][i].to(self.device).detach()
            G_pred_A = G_pred[i].reshape(1,2,256,256).detach()
            G_pred_A = torch.argmax(G_pred_A, dim=1)
            # print(G_pred2.shape,target.shape)
            current_score = self.running_metric.update_cm2(pr=G_pred_A.cpu().numpy(), gt=target.cpu().numpy())
            run_acc_list.append([current_score])
        
        return loss_list,run_acc_list
    def _backward_G(self):
        gt = self.batch['L'].to(self.device).long()

        # torch.argmax(G_pred, dim=1)
        self.G_loss =  self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5*(self._pxl_loss(self.G_middle1, gt)+self._pxl_loss(self.G_middle2, gt))
        # print(self.history_loss['Epoch:'+str(self.epoch_id)])
        self.history_loss['Epoch:'+str(self.epoch_id)] += self.G_loss
        self.G_loss.backward()
    def _backward_G_Test(self,batch):
        gt = batch['L'].to(self.device).long()

        # torch.argmax(G_pred, dim=1)
        G_loss =  self._pxl_loss(self.G_pred1, gt) + self._pxl_loss(self.G_pred2, gt) + 0.5*(self._pxl_loss(self.G_middle1, gt)+self._pxl_loss(self.G_middle2, gt))
        # print(self.history_loss['Epoch:'+str(self.epoch_id)])
        self.history_loss_test['Epoch:'+str(self.epoch_id)] += G_loss
    def iter_val(self):
        self.val = iter(utils.get_loader_rlval(self.args.data_name, img_size=self.args.img_size,
                                  batch_size=self.args.batch_size, is_train=False,
                                  split='trainvalc'))
    def train_models(self):

        self._load_checkpoint()
        loss_dict = {}
        pre_dict = {}
        rewards_dict = {}
        image_policy_dict = {}
        init_loader = utils.get_loader_rlval(self.args.data_name, img_size=self.args.img_size,
                                  batch_size=1, is_train=False,
                                  split='trainval')
        val_dataloader = utils.get_loader_val(self.args.data_name, img_size=self.args.img_size,
                                  batch_size=8, is_train=False,
                                  split='test')
        
        for self.batch_id, batch in enumerate(init_loader, 0):
            B, _, _, c = batch['A'].shape
            for i in range(B):
                pre_dict[batch['name'][i]]=batch['L'][i][0].cpu().numpy()*255
                image_policy_dict[batch['name'][i]] = []
        
        data_augmentation_lis = ['Ours','AugMix','MUM','CutMix','GridMask','MixUp','W/OAug','TokenMix']
        data_augmentation_lis = random.sample(data_augmentation_lis, self.num_actions)
       
        data_augmentation_count = {op:0 for op in data_augmentation_lis}
        self.history_loss = {'Epoch:{}'.format(e):0 for e in range(200)}
        self.history_loss_test = {'Epoch:{}'.format(e):0 for e in range(200)}
        # print
        dataloaders_train = self.dataloaders['train']
        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True

            starttime = time.time()
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            self.logger.write('lr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            # sort
            sorted_losses = sorted(loss_dict.items(), key=lambda x: x[1], reverse=True)

            # # random
            # sorted_losses = list(loss_dict.items())

            # # 打乱列表顺序
            # random.shuffle(sorted_losses)

            # sorted_losses = sorted(loss_dict.items(), key=lambda x: x[1], reverse=False)
            split_rate = min(torch.tanh(torch.FloatTensor([(self.epoch_id)/self.reduct_rates])).item()+0.001,1.0)
            select_len = int(len(sorted_losses)*(1. - split_rate))
            # select_len = int(len(sorted_losses)*(split_rate))
            sorted_losses = sorted_losses[0:select_len]
            print(select_len,split_rate)
            if self.epoch_id != 0:
                f1 = open(os.path.join(self.data_paths), 'w')
                for j in sorted_losses:
                    f1.write(os.path.join(j[0]))
                    f1.write("\n")
                    f1.flush()
                f1.close()
                loss_dict = {}
                self.dataloaders = utils.get_loaders(self.args)
            data_augmentation_count = {op:0 for op in data_augmentation_lis}
            if self.epoch_id>self.warm_ups:
                self.a2c.train()
            for self.batch_id, batch in enumerate(self.dataloaders['train'], 0):
                try:
                    batch_val = next(self.val)
                except (OSError, StopIteration):
                    self.iter_val()
                
                B,_,_,_ = batch['A'].shape
                # print(batch['name'])
                aug1 = []
                aug2 = []
                aug3 = []
                
                if self.epoch_id>self.warm_ups:
                    change_fea = torch.abs(transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['A'])
                                            - transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['B']))

                    state = change_fea
                    state = state.to(self.device)

                    action_probs , v = self.a2c(state)
                    action_dist = Categorical(action_probs) 
                    entropy = action_dist.entropy().mean()
                    action = action_dist.sample()

                    for i in range(B):
                        # reinforment learning
                        action2 = action[i]
                        action2 = data_augmentation_lis[action2]
                        # uniform
                        # action2 = i
                        # random
                        # action2 = random.randint(0,self.num_actions-1)
                        # action2 = data_augmentation_lis[action2]
                        # data_augmentation_count[data_augmentation_lis[action2]] += 1
                        # image_policy_dict[batch['name'][i]].append(action2)
                        data_augmentation_count[action2] += 1
                        if action2 == 'TokenMix':
                            img1_aug, img2_aug, label_aug = apply_augmentation2(batch['A'].to(self.device),batch['B'].to(self.device),batch['L'].to(self.device), i, action2, batch['name'],pre_dict)
                        else:
                            img1_aug, img2_aug, label_aug = apply_augmentation2(batch['A'],batch['B'],batch['L'], i, action2, batch['name'],pre_dict)
                        
                        aug1.append(img1_aug.to(self.device))
                        aug2.append(img2_aug.to(self.device))
                        aug3.append(label_aug.to(self.device))
                    A , B , L = torch.stack(aug1), torch.stack(aug2), torch.stack(aug3)  # change image
                    name = batch['name']
                    A = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(A)
                    B = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(B)

                    batch = {'name': name, 'A': A, 'B': B, 'L': L}


                else:
                    name = batch['name']
                    A = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['A'])
                    B = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['B'])
                    L = batch['L']
                    batch = {'name': name, 'A': A, 'B': B, 'L': L}
                if self.epoch_id>self.warm_ups:
                    requires_grad(self.net_G,False)
                    with torch.no_grad():
                        name = batch_val['name']
                        A = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch_val['A'])
                        B = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch_val['B'])
                        L = batch_val['L']
                        batch_rl1 = {'name': name, 'A': A, 'B': B, 'L': L}
                        cd_loss_beforeaug , run_acc_before= self._forward_pass_loss(batch_rl1)
                    requires_grad(self.net_G,True)

                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()

                if self.epoch_id>self.warm_ups:

                    requires_grad(self.net_G,False)
                    with torch.no_grad():
                        cd_loss_afteraug , runing_acc_after = self._forward_pass_loss(batch_rl1)
                    requires_grad(self.net_G,True)
                    #    Update Actor
                    self.optimizer_a2c.zero_grad()
                    rewards = (100*torch.tensor(runing_acc_after) - 100*torch.tensor(run_acc_before)).to(self.device)
                    if self.batch_id == 0:
                        rewards_dict[self.epoch_id] = 0
                    else:
                        rewards_dict[self.epoch_id] += torch.sum(rewards)
                    advantage = rewards.squeeze() - v.squeeze()

                    actor_loss = -action_dist.log_prob(action) * advantage.detach()

                    critic_loss = advantage.pow(2)

                    a2c_loss = actor_loss.mean() + 0.5 * critic_loss.mean() - 0.001 * entropy
                    # print()
                    a2c_loss.backward()
                    self.optimizer_a2c.step()

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()
            
            
            endtime = time.time()-starttime
            self.logger.write('epoch time: %0.3f \n' % endtime)

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(val_dataloader, 0):
                with torch.no_grad():
                    A = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['A'])
                    B = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['B'])
                    L = batch['L']
                    name = batch['name']
                    batch = {'name': name, 'A': A, 'B': B, 'L': L}
                    self._forward_pass(batch)
                    self._backward_G_Test(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()
            if self.epoch_id!=1111:
                for self.batch_id, batch in enumerate(dataloaders_train, 0):
                    with torch.no_grad():
                        A = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['A'])
                        B = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))(batch['B'])
                        L = batch['L']
                        name = batch['name']
                        batch = {'name': name, 'A': A, 'B': B, 'L': L}
                        self._forward_pass3(batch,loss_dict,pre_dict)
