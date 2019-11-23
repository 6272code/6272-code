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

# Hyper Params
data_name1  = 'go_pro'
data_name2  = 'RESIDE'
data_name3  = 'rain_zhanghe' # DID_MDN
data_name4  = 'DIV'
Pretrained  = './trainedmodel/MBN.pt'

YCRCB_blur  = True
YCRCB_haze  = False
YCRCB_rain  = True
YCRCB_jpeg  = True
jpeg_align_k = 16
bch_size = 1
gpus = 1
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
YOUR_DATA_ROOT = ''
testroot1 = YOUR_DATA_ROOT+data_name1+'/test/'              # GoPro dataset
testroot2 = YOUR_DATA_ROOT+data_name2+'/sots_indoor_test/'  # RESIDE-SOTS
testroot3 = YOUR_DATA_ROOT+data_name3+'/test/' # DID_MDN
testroot4 = YOUR_DATA_ROOT+'LIVE1/'            # LIVE1


# GoPro
test1_list_pth = Home+'lists/'+data_name1+'/blur/test_list.txt'
test1_labels_pth = Home+'lists/'+data_name1+'/label/test_list.txt'

# RESIDE
test2_list_pth = Home+'lists/'+data_name2+'_indoor/sots_test_list.txt'

# Rain_zhanghe
test3_list_pth = Home+'lists/'+data_name3+'/testlist.txt'

# Jpeg
test4_list_pth = Home+'lists/LIVE1/imlist.txt'


# Set transformers
transform = transforms.ToTensor()

# Set convertors
# GoPro
blur_test_cvt  = Convertors.ConvertImageSet_GoPro(testroot1, test1_list_pth, test1_labels_pth,
                                                  transform=transform, crop_size=None,
                                                  with_data_aug=False, resize_to=None)
# Jpeg
jpeg_test_cvt  = Convertors.ConvertImageSet_JpegCompress(testroot4, test4_list_pth,
                                                         crop_size=None, transform=transform,
                                                         with_data_aug=False, Vars=[10], 
                                                         align_k=jpeg_align_k)

# Set data_loaders
blur_testloader = DataLoader(blur_test_cvt, batch_size=1, shuffle=False, num_workers=1)
jpeg_testloader = DataLoader(jpeg_test_cvt, batch_size=1, shuffle=False, num_workers=1) 


# Make net
cleaner = EnDNet(img_dim=3).cuda()
cleaner.load_state_dict(torch.load(Pretrained))
if gpus!= 1:
    cleaner = nn.DataParallel(cleaner, device_ids=range(gpus))
else:
    cleaner = cleaner
cleaner.eval()

with torch.no_grad():
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0       
    for test_iter, data in enumerate(jpeg_testloader):
        img_data, label_data = data        
        img,   img_M,   img_C   = img_data
        label, label_M, label_C = label_data        
    
        img = Variable(img, requires_grad=False).cuda()        
        img_M = Variable(img_M, requires_grad=False).cuda()        
        img_C = Variable(img_C, requires_grad=False).cuda()        
    
        res_C, res_M, res = cleaner(img_C, img_M, img, 'jpeg')
        res = res.data.cpu().numpy()[0]
        res[res>1] = 1
        res[res<0] = 0
        res*= 255
        res = res.astype(np.uint8)
        res = res.transpose((1,2,0))
    
        label = label.numpy()[0]
        label*= 255
        label = label.astype(np.uint8)
        label = label.transpose((1,2,0))
        
        if YCRCB_jpeg:
            res   = cv2.cvtColor(res,   cv2.COLOR_RGB2YCR_CB)[:,:,0]
            label = cv2.cvtColor(label, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            ave_psnr+= psnr(    res, label, data_range=255)
            ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=False)
            ct_num+= 1
        else:
            ave_psnr+= psnr(    res, label, data_range=255)
            ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=True)
            ct_num+= 1
    print('psnr_jpeg: '+str(ave_psnr/float(ct_num))+'.')
    print('ssim_jpeg: '+str(ave_ssim/float(ct_num))+'.')

    
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0       
    for test_iter, te_info in enumerate(open(test3_list_pth).read().splitlines()):
        te_pair_pth = testroot3+te_info
        te_pair = Image.open(te_pair_pth)
        pair_w, pair_h = te_pair.size
    
        img      = te_pair.crop((0, 0, pair_w/2, pair_h))
        te_label = te_pair.crop((pair_w/2, 0, pair_w, pair_h))
    
        img      = np.asarray(img)
        te_label = np.asarray(te_label)
    
        img_M = cv2.resize(img, 
                           (int(img.shape[1]*0.5),
                            int(img.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)
    
        img_C = cv2.resize(img_M, 
                           (int(img_M.shape[1]*0.5),
                            int(img_M.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)
    
        img   = transform(img).unsqueeze(0)
        img_M = transform(img_M).unsqueeze(0)
        img_C = transform(img_C).unsqueeze(0)
    
        img   = Variable(img, requires_grad=False).cuda()
        img_M = Variable(img_M, requires_grad=False).cuda()
        img_C = Variable(img_C, requires_grad=False).cuda()
    
        res = cleaner(img_C, img_M, img, 'rain')
        res = res.data.cpu().numpy()[0]
        res[res>1] = 1
        res[res<0] = 0
        res*= 255
        res = res.astype(np.uint8)
        res = res.transpose((1,2,0))
    
        if YCRCB_rain:
                res      = cv2.cvtColor(res,      cv2.COLOR_RGB2YCR_CB)[:,:,0]
                te_label = cv2.cvtColor(te_label, cv2.COLOR_RGB2YCR_CB)[:,:,0]
                ave_psnr+= psnr(    res, te_label, data_range=255)
                ave_ssim+= ski_ssim(res, te_label, data_range=255, multichannel=False)
                ct_num+= 1
        else:
            ct_num+= 1
            ave_psnr+= psnr(res, te_label, data_range=255)
            ave_ssim+= ski_ssim(res, te_label, data_range=255, multichannel=True)
    
    print('psnr_rain: '+str(ave_psnr/float(ct_num))+'.')
    print('ssim_rain: '+str(ave_ssim/float(ct_num))+'.')

    
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0
    for test_iter, te_info in enumerate(open(test2_list_pth).read().splitlines()):
        te_label_pth = testroot2+'labels/'+te_info.split(' ')[0]        
        te_label = Image.open(te_label_pth)
        te_label = np.asarray(te_label)
    
        for v in np.arange(1, 11, 1):
            te_haze_name = te_info.split('.')[0]+'_'+str(v)+'.png'
            te_haze_pth = testroot2+'images/'+te_haze_name
    
            img = Image.open(te_haze_pth)
            img = np.asarray(img)
            img_M = cv2.resize(img, 
                               (int(img.shape[1]*0.5),
                                int(img.shape[0]*0.5)),
                               interpolation=cv2.INTER_CUBIC)
    
            img_C = cv2.resize(img_M, 
                               (int(img_M.shape[1]*0.5),
                                int(img_M.shape[0]*0.5)),
                               interpolation=cv2.INTER_CUBIC)
    
            img   = transform(img).unsqueeze(0)
            img_M = transform(img_M).unsqueeze(0)
            img_C = transform(img_C).unsqueeze(0)
    
            img   = Variable(img, requires_grad=False).cuda()
            img_M = Variable(img_M, requires_grad=False).cuda()
            img_C = Variable(img_C, requires_grad=False).cuda()
    
            res = cleaner(img_C, img_M, img, 'haze')
            res = res.data.cpu().numpy()[0]
            res[res>1] = 1
            res[res<0] = 0
            res*= 255
            res = res.astype(np.uint8)
            res = res.transpose((1,2,0))
    
            if YCRCB_haze:
                res      = cv2.cvtColor(res,      cv2.COLOR_RGB2YCR_CB)[:,:,0]
                te_label = cv2.cvtColor(te_label, cv2.COLOR_RGB2YCR_CB)[:,:,0]
                ave_psnr+= psnr(    res, te_label, data_range=255)
                ave_ssim+= ski_ssim(res, te_label, data_range=255, multichannel=False)
                ct_num+= 1
            else:
                ave_psnr+= psnr(res, te_label, data_range=255)
                ave_ssim+= ski_ssim(res, te_label, data_range=255, multichannel=True)
                ct_num+= 1
    print('psnr_haze: '+str(ave_psnr/float(ct_num))+'.')
    print('ssim_haze: '+str(ave_ssim/float(ct_num))+'.')
    
    
    ave_psnr = 0.0
    ave_ssim = 0.0
    ct_num = 0
    for test_iter, data in enumerate(blur_testloader):
        img_data, label_data = data        
        img,   img_M,   img_C   = img_data
        label, label_M, label_C = label_data        
    
        img = Variable(img, requires_grad=False).cuda()        
        img_M = Variable(img_M, requires_grad=False).cuda()        
        img_C = Variable(img_C, requires_grad=False).cuda()        
    
        res_C, res_M, res = cleaner(img_C, img_M, img, 'blur')
        res = res.data.cpu().numpy()[0]
        res[res>1] = 1
        res[res<0] = 0
        res*= 255
        res = res.astype(np.uint8)
        res = res.transpose((1,2,0))
    
        label = label.numpy()[0]
        label*= 255
        label = label.astype(np.uint8)
        label = label.transpose((1,2,0))
        
        if YCRCB_blur:
            res   = cv2.cvtColor(res,   cv2.COLOR_RGB2YCR_CB)[:,:,0]
            label = cv2.cvtColor(label, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            ave_psnr+= psnr(    res, label, data_range=255)
            ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=False)
            ct_num+= 1
        else:
            ave_psnr+= psnr(    res, label, data_range=255)
            ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=True)
            ct_num+= 1
    print('psnr_blur: '+str(ave_psnr/float(ct_num))+'.')
    print('ssim_blur: '+str(ave_ssim/float(ct_num))+'.')        
    print('Testing done.')
    


