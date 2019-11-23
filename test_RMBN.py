import os, sys
import torch
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

from pietorch import Home
from pietorch import data_convertors as Convertors
from pietorch import N_modules as n_mods
from pietorch.M_DuRN import EnDNet


def ProcessResult(cleaned_F):
    img_F = cleaned_F.data
    img_F[img_F>1] = 1
    img_F[img_F<0] = 0
    img_M = F.interpolate(img_F, scale_factor=0.5, mode='bicubic')
    img_C = F.interpolate(img_M, scale_factor=0.5, mode='bicubic')
    img_F = Variable(img_F, requires_grad=False).cuda()
    img_M = Variable(img_M, requires_grad=False).cuda()
    img_C = Variable(img_C, requires_grad=False).cuda()
    return img_C, img_M, img_F


# Hyper Params
data_name1 = 'go_pro'
data_name2 = 'RESIDE'
data_name3 = 'rain_zhanghe'
tasks_list = ['jpeg', 'haze', 'blur', 'rain']

YCRCB_blur = True
YCRCB_rain = True
YCRCB_haze = False
YCRCB_jpeg = True
bch_size   = 1
gpus       = 1

Pretrained = './trainedmodel/R-MBN.pt'
m_seed_gpu = 8223752412272754
m_seed_cpu = 8526081014239199321

# Use the seeds. 
if m_seed_gpu is not None and m_seed_cpu is not None:
    if gpus == 1:
        torch.cuda.manual_seed(m_seed_gpu)
    else:
        torch.cuda.manual_seed_all(m_seed_gpu)
    torch.manual_seed(m_seed_cpu)
    print("Use specific seeds.")
else:
    m_seed_gpu = torch.cuda.initial_seed()
    m_seed_cpu = torch.initial_seed()    
   
# Set paths
MY_DATA_ROOT = ''
dataroot1 = MY_DATA_ROOT+data_name1+'/test/'
dataroot2 = MY_DATA_ROOT+data_name2+'/sots_indoor_test/'
dataroot3 = MY_DATA_ROOT+data_name3+'/test/'
dataroot4 = MY_DATA_ROOT+'LIVE1/'

# GoPro
test1_list_pth   = './lists/'+data_name1+'/blur/test_list.txt'
test1_labels_pth = './lists/'+data_name1+'/label/test_list.txt'

test2_list_pth   = './lists/'+data_name2+'_indoor/sots_test_list.txt'  # RESIDE
test3_list_pth   = './lists/'+data_name3+'/testlist.txt'               # Rain_zhanghe
test4_list_pth   = './lists/LIVE1/imlist.txt'                          # Jpeg

# Set transformers
transform = transforms.ToTensor()

# Set convertors
# GoPro
blur_test_cvt = Convertors.ConvertImageSet_GoPro(dataroot1, test1_list_pth, test1_labels_pth,
                                                 transform=transform, crop_size=None,
                                                 with_data_aug=None,  resize_to=None)

jpeg_test_cvt = Convertors.ConvertImageSet_JpegCompress(dataroot4, test4_list_pth, crop_size=None, 
                                                        transform=transform, with_data_aug=None, 
                                                        Vars=[10], align_k=16)

# Set data_loaders
blur_testloader = DataLoader(blur_test_cvt, batch_size=bch_size, shuffle=False, num_workers=1)
jpeg_testloader = DataLoader(jpeg_test_cvt, batch_size=1,        shuffle=False, num_workers=1)


# Make net
cleaner = EnDNet(img_dim=3).cuda()
cleaner.load_state_dict(torch.load(Pretrained))
if gpus!= 1:
    cleaner = nn.DataParallel(cleaner, device_ids=range(gpus))
else:
    cleaner = cleaner
cleaner.eval()

with torch.no_grad():
    # Rain
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0       
    for test_iter, te_info in enumerate(open(test3_list_pth).read().splitlines()):
        te_pair_pth = dataroot3+te_info
        te_pair = Image.open(te_pair_pth)
        pair_w, pair_h = te_pair.size
    
        img_F   = te_pair.crop((0, 0, pair_w/2, pair_h))
        label_F = te_pair.crop((pair_w/2, 0, pair_w, pair_h))
    
        img_F   = np.asarray(img_F)
        label_F = np.asarray(label_F)
    
        img_M = cv2.resize(img_F, 
                           (int(img_F.shape[1]*0.5),
                            int(img_F.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)
    
        img_C = cv2.resize(img_M, 
                           (int(img_M.shape[1]*0.5),
                            int(img_M.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)
    
        img_F = transform(img_F).unsqueeze(0)
        img_M = transform(img_M).unsqueeze(0)
        img_C = transform(img_C).unsqueeze(0)

        img_F = Variable(img_F, requires_grad=False).cuda()
        img_M = Variable(img_M, requires_grad=False).cuda()
        img_C = Variable(img_C, requires_grad=False).cuda()
    
        for j, task in enumerate(tasks_list):
            if task in ['blur', 'jpeg']:
                _,__,clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)
    
            else:
                clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)

        clean_F = clean_F.data.cpu().numpy()[0]
        clean_F[clean_F>1] = 1
        clean_F[clean_F<0] = 0
        clean_F = clean_F*255
        clean_F = clean_F.astype(np.uint8)
        clean_F = clean_F.transpose((1,2,0))

        if YCRCB_rain:
            clean_F = cv2.cvtColor(clean_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            label_F = cv2.cvtColor(label_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=False)
            ct_num+= 1.0
        else:
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=True)
            ct_num+= 1.0
    print('Rain-PSNR: '+str(ave_psnr/ct_num)+'.')
    print('Rain-SSIM: '+str(ave_ssim/ct_num)+'.')


    # Jpeg
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num   = 0
    for test_iter, data in enumerate(jpeg_testloader):
        img_data, label_data = data
        img_F,   img_M,   img_C   = img_data
        label_F, label_M, label_C = label_data        
    
        img_F = Variable(img_F, requires_grad=False).cuda()
        img_M = Variable(img_M, requires_grad=False).cuda()
        img_C = Variable(img_C, requires_grad=False).cuda()
    
        for j, task in enumerate(tasks_list):
            if task in ['blur', 'jpeg']:
                _,__,clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)
    
            else:
                clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)

        clean_F = clean_F.data.cpu().numpy()[0]
        clean_F[clean_F>1] = 1
        clean_F[clean_F<0] = 0
        clean_F = clean_F*255
        clean_F = clean_F.astype(np.uint8)
        clean_F = clean_F.transpose((1,2,0))

        label_F = label_F.numpy()[0]
        label_F = label_F*255
        label_F = label_F.astype(np.uint8)
        label_F = label_F.transpose((1,2,0))

        if YCRCB_jpeg:
            clean_F = cv2.cvtColor(clean_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            label_F = cv2.cvtColor(label_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=False)
            ct_num+= 1.0
        else:
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=True)
            ct_num+= 1.0
    print('Jpeg-PSNR: '+str(ave_psnr/ct_num)+'.')
    print('Jpeg-SSIM: '+str(ave_ssim/ct_num)+'.')
    
    # Blur
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num   = 0
    for test_iter, data in enumerate(blur_testloader):
        img_data, label_data = data
        img_F,   img_M,   img_C   = img_data
        label_F, label_M, label_C = label_data        
    
        img_F = Variable(img_F, requires_grad=False).cuda()
        img_M = Variable(img_M, requires_grad=False).cuda()
        img_C = Variable(img_C, requires_grad=False).cuda()
    
        for j, task in enumerate(tasks_list):
            if task in ['blur', 'jpeg']:
                _,__,clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)
    
            else:
                clean_F = cleaner(img_C, img_M, img_F, task)
                img_C, img_M, img_F = ProcessResult(clean_F)

        clean_F = clean_F.data.cpu().numpy()[0]
        clean_F[clean_F>1] = 1
        clean_F[clean_F<0] = 0
        clean_F = clean_F*255
        clean_F = clean_F.astype(np.uint8)
        clean_F = clean_F.transpose((1,2,0))

        label_F = label_F.numpy()[0]
        label_F = label_F*255
        label_F = label_F.astype(np.uint8)
        label_F = label_F.transpose((1,2,0))

        if YCRCB_blur:
            clean_F = cv2.cvtColor(clean_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            label_F = cv2.cvtColor(label_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=False)
            ct_num+= 1.0
        else:
            ave_psnr+= psnr(    clean_F, label_F, data_range=255)
            ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=True)
            ct_num+= 1.0
    print('Blur-PSNR: '+str(ave_psnr/ct_num)+'.')
    print('Blur-SSIM: '+str(ave_ssim/ct_num)+'.')

    
    # Haze
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0
    for test_iter, te_info in enumerate(open(test2_list_pth).read().splitlines()):
        te_label_pth = dataroot2+'labels/'+te_info.split(' ')[0]        
        label_F = Image.open(te_label_pth)
        label_F = np.asarray(label_F)
        label_F = align_to_k(label_F, 16)

        for v in np.arange(1, 11, 1):
            te_haze_name = te_info.split('.')[0]+'_'+str(v)+'.png'
            te_haze_pth = dataroot2+'images/'+te_haze_name
            img_F = Image.open(te_haze_pth)
            img_F = np.asarray(img_F)
            img_F = align_to_k(img_F, 16)
            img_M = cv2.resize(img_F, 
                               (int(img_F.shape[1]*0.5),
                                int(img_F.shape[0]*0.5)),
                               interpolation=cv2.INTER_CUBIC)
            img_C = cv2.resize(img_M, 
                               (int(img_M.shape[1]*0.5),
                                int(img_M.shape[0]*0.5)),
                               interpolation=cv2.INTER_CUBIC)

            img_F = transform(img_F).unsqueeze(0)
            img_M = transform(img_M).unsqueeze(0)
            img_C = transform(img_C).unsqueeze(0)
            img_F = Variable(img_F, requires_grad=False).cuda()
            img_M = Variable(img_M, requires_grad=False).cuda()
            img_C = Variable(img_C, requires_grad=False).cuda()
        
            for j, task in enumerate(tasks_list):
                if task in ['blur', 'jpeg']:
                    _,__,clean_F = cleaner(img_C, img_M, img_F, task)
                    img_C, img_M, img_F = ProcessResult(clean_F)
        
                else:
                    clean_F = cleaner(img_C, img_M, img_F, task)
                    img_C, img_M, img_F = ProcessResult(clean_F)
    
            clean_F = clean_F.data.cpu().numpy()[0]
            clean_F[clean_F>1] = 1
            clean_F[clean_F<0] = 0
            clean_F = clean_F*255
            clean_F = clean_F.astype(np.uint8)
            clean_F = clean_F.transpose((1,2,0))
    
            if YCRCB_haze:
                clean_F = cv2.cvtColor(clean_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
                label_F = cv2.cvtColor(label_F, cv2.COLOR_RGB2YCR_CB)[:,:,0]
                ave_psnr+= psnr(    clean_F, label_F, data_range=255)
                ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=False)
                ct_num+= 1.0
            else:
                ave_psnr+= psnr(    clean_F, label_F, data_range=255)
                ave_ssim+= ski_ssim(clean_F, label_F, data_range=255, multichannel=True)
                ct_num+= 1.0
    print('Haze-PSNR: '+str(ave_psnr/ct_num)+'.')
    print('Haze-SSIM: '+str(ave_ssim/ct_num)+'.')
        
