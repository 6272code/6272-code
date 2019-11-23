import numpy as np
import torch
import random
import torchvision
import math
import torch.nn as nn
import itertools
import skimage as ski
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
from scipy import ndimage
from scipy.special import gamma
from skimage.transform import warp
import cv2
import h5py
from .utils import GaussianNoise, SaltAndPepper, GaussianBlur, align_to_k

# Loading data in "caffe" style.
# Given: i) image_root. ii) image_list_path.
# Obtain: Images in the listed order.
#------------------------------------------

# Gaussian noise data convertor
class ConvertImageSet(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert("RGB")

    
    def __init__(self, root, imlist_pth, transform=None):
        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform

    # process data
    def __getitem__(self, index):
        im_pth = self.imlist[index]
        img = self.im_loader(self.root+im_pth)

        # transform the image (or not)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)        


# real-world noise data (by HkPoly) convertor
class ConvertImageSet_RealHK(data.Dataset):
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist
        
    # image reader
    def im_loader(self, im_pth):
        return Image.open(im_pth)

    def __init__(self, root, imlist_pth, phase, with_data_aug, crop_size=None, transform=None):
        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform = transform
        self.crop_size = crop_size
        self.phase = phase
        self.with_data_aug = with_data_aug
        
    # process data
    def __getitem__(self, index):
        im_name = self.imlist[index]
        label = self.im_loader(self.root+im_name)
        if self.phase == 'train':
            noisy = self.im_loader(self.root+im_name.split('mean')[0]+'Real.JPG')
        elif self.phase == 'test':
            noisy = self.im_loader(self.root+im_name.split('mean')[0]+'real.PNG')

        # crop the image (or not)
        if self.crop_size!= None:
            W, H = label.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
            noisy = noisy.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
        else:
            pass

        # Do augmentation (or not)
        if self.with_data_aug:            
            # Horizontal flip
            if random.random() > 0.5:                
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                noisy = noisy.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
                noisy = noisy.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                label = label.rotate(angle)
                noisy = noisy.rotate(angle)                
        else:
            pass

        # transform the image (or not)
        if self.transform is not None:
            label = self.transform(label)
            noisy = self.transform(noisy)

        return noisy, label

    def __len__(self):
        return len(self.imlist)


# Raindrop removal data convertor
class ConvertImageSet_RainDrop(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist    

    # image loader
    def im_loader(self, im_pth):
#        return cv2.imread(im_pth)
#        return cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB)
        return np.asarray(Image.open(im_pth).convert('RGB'))

    # correct image size
    def align_to_four(self, img):
        a_row = int(img.shape[0]/4)*4
        a_col = int(img.shape[1]/4)*4
        img = img[0:a_row, 0:a_col]
        return img
    
    def __init__(self, root, imlist_pth, crop_size=256, houzhui='png', transform=None):
        self.root = root        
        self.houzhui = houzhui  # file extension (such as .png)
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.crop_size = crop_size
        self.img_chunk = []
        self.label_chunk = []
        for i, im_name in enumerate(self.imlist):
            label = self.im_loader(self.root+'gt/'+im_name+'_clean.'+self.houzhui)

            self.label_chunk.append(Image.fromarray(self.align_to_four(label.copy())))
            img = self.im_loader(self.root+'data/'+im_name+'_rain.'+self.houzhui)
            self.img_chunk.append(Image.fromarray(self.align_to_four(img.copy())))

            if i%100 == 99 or i == len(self.imlist)-1:
                print('loading rain_drop image: '+str(i+1)+'/'+str(len(self.imlist)))
            
        
    # process data
    def __getitem__(self, index):
        label = self.label_chunk[index]
        img = self.img_chunk[index]

        # crop the image (or not)
        if self.crop_size!= None:
            W, H = label.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
            img = img.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))

        else:
            pass

        # transform the image (or not)
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
            
        return img, label

    def __len__(self):
        return len(self.imlist)


# Rain streak removal data convertor -- the Rainy Dataset (with guided image filtering)
# This convertor will be used only when you want to try the DDN (CVPR'17)
class ConvertImageSet_RainyDataWithFilter(data.Dataset):    
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        # There is an annoying non-RGB image in the dataset.
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, label_root, rainy_root, imlist_pth, crop_size, transform, r):
        self.label_root = label_root
        self.rainy_root = rainy_root
        self.crop_size = crop_size

        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        
        # Vars: 14 types of rain streaks.
        self.Vars = np.arange(1, 15, 1)

        # r: width for guided image filtering.
        # DDN's original setting is 15.
        self.r = r


    # Guided Image Filtering implementation.
    def filter_image(self, image, rad):
        r = rad
        data_ = np.copy(np.asarray(image))
        H, W, C =  data_.shape
        eps = 1.0
        batch_q = np.zeros((1, H, W, C))
        data = data_/255.0

        for j in range(3):
            I = data[:, :,j]
            p = data[:, :,j]
            ones_array = np.ones([H, W])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType=0)/N
            q = mean_a * I + mean_b
            batch_q[0, :, :,j] = q

        low_freq = batch_q[0]
        high_freq = data - low_freq

        return high_freq, low_freq

    # process data         
    def __getitem__(self, index):
        img_id, img_target = self.imlist[index]
        
        # prepare label data
        label = self.im_loader(self.label_root+img_id)        
        W, H = label.size

        # random crop the ground truth image
        x_offset = random.randint(0, W - self.crop_size)
        y_offset = random.randint(0, H - self.crop_size)
        label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))

        # guided-image-filtering the ground truth image.
        label_high_freq, label_low_freq = self.filter_image(label, self.r)
        label_high_freq = torch.FloatTensor(label_high_freq.transpose((2,0,1)))
        label_low_freq = torch.FloatTensor(label_low_freq.transpose((2,0,1)))
        label = self.transform(label)

        
        # prepare rainy data
        # change the [1,1] to [1,0] if you want to mix rain-free patches into input.
        if random.choice([1,1]):
            img_name = img_id.split('.')[0]+'_'+str(random.choice(self.Vars))+'.jpg'
            img_pth = self.rainy_root+img_name
        else:
            img_pth = self.label_root+img_id
        img = self.im_loader(img_pth)
        
        # crop the rainy image
        img = img.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))

        # guided-image-filtering the rainy image
        high_freq, low_freq = self.filter_image(img, self.r)
        high_freq = torch.FloatTensor(high_freq.transpose((2,0,1)))
        low_freq = torch.FloatTensor(low_freq.transpose((2,0,1)))
        img = self.transform(img)

        return (high_freq, low_freq, img), (label_high_freq, label_low_freq, label)

    def __len__(self):
        return len(self.imlist)

        

# Rain streak removal convertor --- the Rainy Dataset (DDN data)
class ConvertImageSet_RainyData(data.Dataset):    
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        # There is an annoying non-RGB image in the dataset.
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, dataroot, imlist_pth, crop_size, transform, with_data_aug):
        
        self.label_root = dataroot+'label/'
        self.rainy_root = dataroot+'rain_image/'
        self.crop_size = crop_size
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.with_data_aug = with_data_aug
        
        # Vars: 14 types of rain streaks.
        self.Vars = np.arange(1, 15, 1)
        
    # process data         
    def __getitem__(self, index):
        img_id = self.imlist[index]
        
        # prepare label data
        label = self.im_loader(self.label_root+img_id)        
        W, H = label.size

        # random crop the ground truth image
        x_offset = random.randint(0, W - self.crop_size)
        y_offset = random.randint(0, H - self.crop_size)
        label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
        
        # prepare rainy data
        # change the [1,1] to [1,0] if you want to mix rain-free patches into input.
        if random.choice([1,1]):
            img_name = img_id.split('.')[0]+'_'+str(random.choice(self.Vars))+'.jpg'
            img_pth = self.rainy_root+img_name
        else:
            img_pth = self.label_root+img_id
        rainy = self.im_loader(img_pth)
        
        # crop the rainy image
        rainy = rainy.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                rainy = rainy.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
                rainy = rainy.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                label = label.rotate(angle)
                rainy = rainy.rotate(angle)                

        rainy = self.transform(rainy)
        label = self.transform(label)

        return rainy, label
                

    def __len__(self):
        return len(self.imlist)



# Rain streak removal convertor --- the D-Rainy Dataset (DID-MDN data)
class ConvertImageSet_D_RainyData(data.Dataset):
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        # There is an annoying non-RGB image in the dataset.
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, dataroot, imlist_pth, crop_size, transform, with_data_aug):

        self.dataroot = dataroot
        self.crop_size = crop_size
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.with_data_aug = with_data_aug
        self.Vars = ['Rain_Heavy', 'Rain_Medium', 'Rain_Light']
        self.images_chunk = []
        self.labels_chunk = []
        for name in self.imlist:
            var_chunk = []
            for var in self.Vars:
                pair = self.im_loader(self.dataroot+var+'/train2018new/'+name)
                pair_w, pair_h = pair.size
                rainy = pair.crop((0, 0, pair_w/2, pair_h))
                label = pair.crop((pair_w/2, 0, pair_w, pair_h))
                var_chunk.append(rainy)
            self.labels_chunk.append(label)
            self.images_chunk.append(var_chunk)
        
    # process data         
    def __getitem__(self, index):        
        label = self.labels_chunk[index]
        rainy = self.images_chunk[index][random.choice([0,1,2])]

        if self.crop_size!= None:
            W, H = label.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            rainy = rainy.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
            label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                rainy = rainy.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
#            if random.random() > 0.5:
#                label = label.transpose(Image.FLIP_TOP_BOTTOM)
#                rainy = rainy.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
#            if random.random() > 0.5:
#                angle = random.choice([90, 180, 270])
#                label = label.rotate(angle)
#                rainy = rainy.rotate(angle)                
                

        label = np.asarray(label)
        rainy = np.asarray(rainy)            
        rainy_M = cv2.resize(rainy, 
                             (int(rainy.shape[1]*0.5), 
                              int(rainy.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        rainy_C = cv2.resize(rainy_M, 
                             (int(rainy_M.shape[1]*0.5), 
                              int(rainy_M.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        label_M = cv2.resize(label, 
                             (int(label.shape[1]*0.5), 
                              int(label.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        label_C = cv2.resize(label_M, 
                             (int(label_M.shape[1]*0.5), 
                              int(label_M.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        rainy   = self.transform(rainy)
        rainy_M = self.transform(rainy_M)
        rainy_C = self.transform(rainy_C)

        label   = self.transform(label)
        label_M = self.transform(label_M)
        label_C = self.transform(label_C)

        return (rainy, rainy_M, rainy_C), (label, label_M, label_C)
                

    def __len__(self):
        return len(self.imlist)


        
# motion-blur removal convertor --- the GoPro Dataset
class ConvertImageSet_GoPro(data.Dataset):
    # txt reader 
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
#        return Image.open(im_pth)
        return cv2.imread(im_pth)

    
    def __init__(self, dataroot, imlist_pth, labels_pth, transform, crop_size, with_data_aug,
                 resize_to=None, mode='train'):
        
        self.dataroot   = dataroot
        self.resize_to  = resize_to
        self.imlist     = self.flist_reader(imlist_pth)
        self.label_list = self.flist_reader(labels_pth)

        self.transform = transform
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.mode = mode
        self.labels_chunk = []        
        self.images_chunk = []

        for i, label_name in enumerate(self.label_list):
            blur_name = self.imlist[i]
            self.labels_chunk.append(self.im_loader(self.dataroot+label_name))
            self.images_chunk.append(self.im_loader(self.dataroot+ blur_name))

    # process data                                                                       
    def __getitem__(self, index):
        blur_img  = self.images_chunk[index]
        label_img = self.labels_chunk[index]       
        if type(self.resize_to).__name__ == 'tuple':
            blur_img  = cv2.resize(blur_img,  self.resize_to)
            label_img = cv2.resize(label_img, self.resize_to)
        else:
            pass
        
        blur_img  = cv2.cvtColor(blur_img,  cv2.COLOR_BGR2RGB)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        if self.crop_size!= None:
            blur_img  = Image.fromarray(blur_img)
            label_img = Image.fromarray(label_img)

            W, H = label_img.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            label_img = label_img.crop((x_offset, 
                                        y_offset, 
                                        x_offset+self.crop_size, 
                                        y_offset+self.crop_size))

            blur_img= blur_img.crop((x_offset, 
                                     y_offset, 
                                     x_offset+self.crop_size, 
                                     y_offset+self.crop_size))

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                blur_img = blur_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
#            if random.random() > 0.5:
#                label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)
#                blur_img = blur_img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
#            if random.random() > 0.5:
#                angle = random.choice([90, 180, 270])
#                label_img = label_img.rotate(angle)
#                blur_img = blur_img.rotate(angle)              

        label_img = np.asarray(label_img)
        blur_img = np.asarray(blur_img)
        
        blur_img_M = cv2.resize(blur_img, 
                                (int(blur_img.shape[1]*0.5), 
                                 int(blur_img.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        blur_img_C = cv2.resize(blur_img_M, 
                                (int(blur_img_M.shape[1]*0.5), 
                                 int(blur_img_M.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        label_img_M = cv2.resize(label_img, 
                                (int(label_img.shape[1]*0.5), 
                                 int(label_img.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        label_img_C = cv2.resize(label_img_M, 
                                (int(label_img_M.shape[1]*0.5), 
                                 int(label_img_M.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        blur_img   = self.transform(blur_img)
        blur_img_M = self.transform(blur_img_M)
        blur_img_C = self.transform(blur_img_C)

        label_img   = self.transform(label_img)
        label_img_M = self.transform(label_img_M)
        label_img_C = self.transform(label_img_C)

        return (blur_img, blur_img_M, blur_img_C), (label_img, label_img_M, label_img_C)


    def __len__(self):
        return len(self.imlist)


        
# haze removal convertor --- the RESIDE
class ConvertImageSet_RESIDE(data.Dataset):
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist
        
    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth)

    def __init__(self, root, imlist_pth, crop_size=256, transform=None, with_data_aug=True):
        self.root = root
        self.with_data_aug = with_data_aug
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.crop_size = crop_size

        self.labels_chunk = []
        self.images_chunk = []
        for name in self.imlist:
            self.labels_chunk.append(self.im_loader(self.root+'labels/'+name))
            var_chunk = []
            for var in np.arange(1, 11, 1):
                var_chunk.append(self.im_loader(self.root+'images/'+name.split('.')[0]+
                                                        '_'+str(var)+'.png'))
            self.images_chunk.append(var_chunk)
                
    # process data                                                                       
    def __getitem__(self, index):
        label = self.labels_chunk[index]
        img   = self.images_chunk[index][random.choice(np.arange(0, 10, 1))]

        # random crop the ground truth image
        W, H = label.size
        x_offset = random.randint(0, W - self.crop_size)
        y_offset = random.randint(0, H - self.crop_size)
        label = label.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))
        
        # crop the hazy image 
        img = img.crop((x_offset, y_offset, x_offset+self.crop_size, y_offset+self.crop_size))


        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
                img   = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                label = label.rotate(angle)
                img   = img.rotate(angle)              

            label = np.asarray(label)
            img   = np.asarray(img)

        img_M = cv2.resize(img, 
                           (int(img.shape[1]*0.5), 
                            int(img.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)

        img_C = cv2.resize(img_M, 
                           (int(img_M.shape[1]*0.5), 
                            int(img_M.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)

        label_M = cv2.resize(label, 
                             (int(label.shape[1]*0.5), 
                              int(label.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        label_C = cv2.resize(label_M, 
                             (int(label_M.shape[1]*0.5), 
                              int(label_M.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        img   = self.transform(img)
        img_M = self.transform(img_M)
        img_C = self.transform(img_C)

        label   = self.transform(label)
        label_M = self.transform(label_M)
        label_C = self.transform(label_C)

        return (img, img_M, img_C), (label, label_M, label_C)

    def __len__(self):
        return len(self.imlist)

        
        
# haze removal convertor --- the Dehaze Dataset (DCPDN, CVPR'18)
# The images had been saved in .h5 files by the author (Zhang).
class ConvertImageSet_zhanghe(data.Dataset):
    # txt reader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist
        
    # image (.h5 file) loader
    def im_loader(self, im_pth, resize_to=None):
        f = h5py.File(im_pth, 'r')
        keys = f.keys()
        label = np.asarray(f[keys[1]])
        img = np.asarray(f[keys[2]])
        if label.max() > 1 or img.max() > 1:
            raise Exception('Data out of range [0~1]')

        # resize it (or not)
        if resize_to!= None:
            label = label*255
            label = label.astype(np.uint8)
            label = np.asarray(Image.fromarray(label).resize(resize_to, Image.BICUBIC))
            label = label/255.0

            img = img*255
            img = img.astype(np.uint8)
            img = np.asarray(Image.fromarray(img).resize(resize_to, Image.BICUBIC))
            img = img/255.0
            return (img, label)
            
        else:
            return (img, label)

    def __init__(self, root, imlist_pth, crop_size=64, resize_to=None):
        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.crop_size = crop_size
        self.resize_to = resize_to

    # process data                                                                       
    def __getitem__(self, index):
        im_pth = self.imlist[index]
        img, label = self.im_loader(self.root+im_pth, resize_to=self.resize_to)

        # crop them (or not)
        if not self.crop_size is None:
            H, W, _ = label.shape
            h_offset = random.randint(0, H - self.crop_size)
            w_offset = random.randint(0, W - self.crop_size)

            img = img[h_offset:h_offset+self.crop_size, w_offset:w_offset+self.crop_size, :]
            label = label[h_offset:h_offset+self.crop_size, w_offset:w_offset+self.crop_size, :]
        else:
            pass

        img = img.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        img = torch.FloatTensor(img)
        label = torch.FloatTensor(label)

#        return img, label, im_pth
        return img, label

    def __len__(self):
        return len(self.imlist)
        
       
class ConvertImageSet_JpegCompress(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth, k=4):
        img = np.asarray(Image.open(im_pth).convert("RGB"))
        if not k is None:
            img = align_to_k(img, k)
        return Image.fromarray(img)

    
    def __init__(self, root, imlist_pth, crop_size=256, Vars=[10, 20, 30, 40], transform=None, with_data_aug=False, align_k=None):
        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.Vars = Vars
        self.crop_size=crop_size
        self.with_data_aug = with_data_aug
        self.align_k = align_k

        self.images_chunk = []
        for name in self.imlist:
            self.images_chunk.append(self.im_loader(self.root+name, self.align_k))

    # process data
    def __getitem__(self, index):
        img = self.images_chunk[index]

        if not self.crop_size is None:
            W, H = img.size
            w_offset = random.randint(0, W - self.crop_size)
            h_offset = random.randint(0, H - self.crop_size)
            img = img.crop((w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size))
        else:
            pass

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
#            # Vertical flip
#            if random.random() > 0.5:
#                img = img.transpose(Image.FLIP_TOP_BOTTOM)
#            # Random rotation (90 or 180 or 270)
#            if random.random() > 0.5:
#                angle = random.choice([90, 180, 270])
#                img = img.rotate(angle)
        else:
            pass            

        buff = BytesIO()

        comp_img = img.copy()
        comp_img.save(buff, "JPEG", quality=random.choice(self.Vars))
        comp_img = Image.open(buff)

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                comp_img = comp_img.transpose(Image.FLIP_LEFT_RIGHT)                

        img      = np.asarray(img)
        comp_img = np.asarray(comp_img)            
        comp_img_M = cv2.resize(comp_img, 
                             (int(comp_img.shape[1]*0.5), 
                              int(comp_img.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        comp_img_C = cv2.resize(comp_img_M, 
                             (int(comp_img_M.shape[1]*0.5), 
                              int(comp_img_M.shape[0]*0.5)),
                             interpolation=cv2.INTER_CUBIC)

        img_M = cv2.resize(img, 
                           (int(img.shape[1]*0.5), 
                            int(img.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)

        img_C = cv2.resize(img_M, 
                           (int(img_M.shape[1]*0.5), 
                            int(img_M.shape[0]*0.5)),
                           interpolation=cv2.INTER_CUBIC)
        
        # transform the image (or not)
        if self.transform is not None:
            comp_img   = self.transform(comp_img)
            comp_img_M = self.transform(comp_img_M)
            comp_img_C = self.transform(comp_img_C)

            img = self.transform(img)
            img_M = self.transform(img_M)
            img_C = self.transform(img_C)

        return (comp_img, comp_img_M, comp_img_C), (img, img_M, img_C)

    def __len__(self):
        return len(self.imlist)        


class ConvertImageSet_NoiseOfTwo(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, root, imlist_pth, crop_size=256, 
                 G_stds=np.arange(1, 51, 1), 
                 St_probs=np.arange(0.01, 0.31, 0.01),
                 transform=None,
                 with_data_aug=False):

        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.G_stds = G_stds
        self.St_probs = St_probs
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.data_chunk = []
        for i, im_name in enumerate(self.imlist):
            img = self.im_loader(self.root+im_name)
            self.data_chunk.append(img.copy())

            if i%100 == 99 or i == len(self.imlist)-1:
                print('loading noise image: '+str(i+1)+'/'+str(len(self.imlist)))


    # process data
    def __getitem__(self, index):

        img = self.data_chunk[index]
        if not self.crop_size is None:
            W, H = img.size
            w_offset = random.randint(0, W - self.crop_size)
            h_offset = random.randint(0, H - self.crop_size)
            img = img.crop((w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size))
        else:
            pass

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
        else:
            pass            

        img = self.transform(img)
        label = img.clone()
        if random.choice([0,1]):
            img = GaussianNoise(img, self.G_stds)
        else:
            img = SaltAndPepper(img, self.St_probs)

        return img, label

    def __len__(self):
        return len(self.imlist)        



class ConvertImageSet_GaussianNoise(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, root, imlist_pth, crop_size=64, 
                 noise_vars=np.arange(0, 55, 5), 
                 transform=None,
                 with_data_aug=False):

        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.G_stds = noise_vars
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.data_chunk = []
        for i, im_name in enumerate(self.imlist):
            img = self.im_loader(self.root+im_name)
            self.data_chunk.append(img.copy())

            if i%100 == 99 or i == len(self.imlist)-1:
                print('loading noise image: '+str(i+1)+'/'+str(len(self.imlist)))


    # process data
    def __getitem__(self, index):

        img = self.data_chunk[index]
        if not self.crop_size is None:
            W, H = img.size
            w_offset = random.randint(0, W - self.crop_size)
            h_offset = random.randint(0, H - self.crop_size)
            img = img.crop((w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size))
        else:
            pass

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
        else:
            pass            

        img = self.transform(img)
        label = img.clone()
        img = GaussianNoise(img, self.G_stds)
        
        return img, label

    def __len__(self):
        return len(self.imlist)        



class ConvertImageSet_GaussianBlur(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, root, imlist_pth, crop_size=64, 
                 blur_vars=np.arange(0, 5.5, 0.5), 
                 transform=None,
                 with_data_aug=False):

        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.G_stds = blur_vars
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.data_chunk = []
        for i, im_name in enumerate(self.imlist):
            img = self.im_loader(self.root+im_name)
            self.data_chunk.append(img.copy())

            if i%100 == 99 or i == len(self.imlist)-1:
                print('loading blur image: '+str(i+1)+'/'+str(len(self.imlist)))


    # process data
    def __getitem__(self, index):

        img = self.data_chunk[index]
        if not self.crop_size is None:
            W, H = img.size
            w_offset = random.randint(0, W - self.crop_size)
            h_offset = random.randint(0, H - self.crop_size)
            img = img.crop((w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size))
        else:
            pass

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
        else:
            pass            

        
        label = img.copy()
        label = self.transform(label)

        img = GaussianBlur(img, self.G_stds)
        img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.imlist)        


class ConvertImageSet_MixSyn(data.Dataset):
    # txt loader
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert("RGB")
    
    def __init__(self, root, imlist_pth, crop_size=64,                 
                 blur_vars=np.arange(0, 5.5, 0.5),
                 noise_vars=np.arange(0, 55, 5),
                 jpeg_vars=[10, 20, 30, 40],
                 transform=None,
                 with_data_aug=False):

        self.root = root
        self.imlist = self.flist_reader(imlist_pth)
        self.transform=transform
        self.B_vars = blur_vars
        self.N_vars = noise_vars
        self.J_vars = jpeg_vars
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.data_chunk = []
        for i, im_name in enumerate(self.imlist):
            img = self.im_loader(self.root+im_name)
            self.data_chunk.append(img.copy())

            if i%100 == 99 or i == len(self.imlist)-1:
                print('loading blur image: '+str(i+1)+'/'+str(len(self.imlist)))

    # process data
    def __getitem__(self, index):

        img = self.data_chunk[index]
        if not self.crop_size is None:
            W, H = img.size
            w_offset = random.randint(0, W - self.crop_size)
            h_offset = random.randint(0, H - self.crop_size)
            img = img.crop((w_offset, h_offset, w_offset+self.crop_size, h_offset+self.crop_size))
        else:
            pass

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
        else:
            pass            

        label = img.copy()
        label = self.transform(label)        
        img = GaussianBlur(img, self.B_vars)
        img = self.transform(img)
        img = GaussianNoise(img, self.N_vars)
        img = img.numpy().transpose((1,2,0))
        img[img>1] = 1
        img[img<0] = 0
        img = img*255
        img = Image.fromarray(img.astype(np.uint8))
        buff = StringIO.StringIO()
        img.save(buff, 'JPEG', quality=random.choice(self.J_vars))
        img = Image.open(buff)
        img = self.transform(img)
        
        return img, label

    def __len__(self):
        return len(self.imlist)        
        


class ConvertImageSet_RealRain(data.Dataset):
    # txt reader 
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l.split(' '))
        return imlist

    # image loader
    def im_loader(self, im_pth):
        return Image.open(im_pth).convert('RGB')
#        return cv2.imread(im_pth)

    
    def __init__(self, dataroot, imlist_pth, transform, crop_size, with_data_aug,
                 resize_to=None):
        
        self.dataroot   = dataroot
        self.resize_to  = resize_to
        self.imlist     = self.flist_reader(imlist_pth)

        self.transform = transform
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.labels_chunk = []        
        self.images_chunk = []

        for i, samp_info in enumerate(self.imlist):
            rainy_name, label_name = samp_info
            label = self.im_loader(self.dataroot+label_name)
            rainy = self.im_loader(self.dataroot+rainy_name)            
            self.labels_chunk.append(np.asarray(label))
            self.images_chunk.append(np.asarray(rainy))

    # process data                                                                       
    def __getitem__(self, index):
        rainy_img = self.images_chunk[index]
        label_img = self.labels_chunk[index]       
        if type(self.resize_to).__name__ == 'tuple':
            quit()
#            rainy_img = cv2.resize(rainy_img, self.resize_to)
#            label_img = cv2.resize(label_img, self.resize_to)
        else:
            pass
        
#        rainy_img = cv2.cvtColor(rainy_img, cv2.COLOR_BGR2RGB)
#        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        if self.crop_size!= None:
            rainy_img = Image.fromarray(rainy_img)
            label_img = Image.fromarray(label_img)

            W, H = label_img.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            label_img = label_img.crop((x_offset, 
                                        y_offset, 
                                        x_offset+self.crop_size, 
                                        y_offset+self.crop_size))

            rainy_img = rainy_img.crop((x_offset, 
                                        y_offset, 
                                        x_offset+self.crop_size, 
                                        y_offset+self.crop_size))

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                rainy_img = rainy_img.transpose(Image.FLIP_LEFT_RIGHT)

            # Vertical flip
#            if random.random() > 0.5:
#                label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)
#                blur_img = blur_img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
#            if random.random() > 0.5:
#                angle = random.choice([90, 180, 270])
#                label_img = label_img.rotate(angle)
#                blur_img = blur_img.rotate(angle)              

        label_img = np.asarray(label_img)
        rainy_img = np.asarray(rainy_img)
        
        rainy_img_M = cv2.resize(rainy_img, 
                                 (int(rainy_img.shape[1]*0.5), 
                                  int(rainy_img.shape[0]*0.5)),
                                 interpolation=cv2.INTER_CUBIC)

        rainy_img_C = cv2.resize(rainy_img_M, 
                                 (int(rainy_img_M.shape[1]*0.5), 
                                  int(rainy_img_M.shape[0]*0.5)),
                                 interpolation=cv2.INTER_CUBIC)

        label_img_M = cv2.resize(label_img, 
                                (int(label_img.shape[1]*0.5), 
                                 int(label_img.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        label_img_C = cv2.resize(label_img_M, 
                                (int(label_img_M.shape[1]*0.5), 
                                 int(label_img_M.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        rainy_img   = self.transform(rainy_img)
        rainy_img_M = self.transform(rainy_img_M)
        rainy_img_C = self.transform(rainy_img_C)

        label_img   = self.transform(label_img)
        label_img_M = self.transform(label_img_M)
        label_img_C = self.transform(label_img_C)

        return (rainy_img, rainy_img_M, rainy_img_C), (label_img, label_img_M, label_img_C)


    def __len__(self):
        return len(self.labels_chunk)
        



class ConvertImageSet_RESIDE_beta_mytrain(data.Dataset):
    # txt reader 
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    # image loader
    def im_loader(self, im_pth):
        img = Image.open(im_pth).convert('RGB')
        w,h = img.size
        if not self.crop_size is None:
            if w < self.crop_size or h < self.crop_size:
                re_w = max(w, self.crop_size)
                re_h = max(h, self.crop_size)
                img = img.resize((re_w, re_h), resample=Image.BICUBIC)

        if not self.align_k is None:
            img = np.asarray(img)
            img = align_to_k(img, self.align_k)
            img = Image.fromarray(img)

        img = np.asarray(img)
        return img
            
#        return Image.open(im_pth)
#        return cv2.imread(im_pth)
        

    
    def __init__(self, dataroot, imlist_pth, label_pth, transform, crop_size, with_data_aug,
                 resize_to=None, align_k=None):
        
        self.dataroot   = dataroot
        self.resize_to  = resize_to
        self.imlist     = self.flist_reader(imlist_pth)
        self.labellist  = self.flist_reader(label_pth )
        self.align_k    = align_k
#        random.shuffle(self.imlist)
#        random.shuffle(self.imlist)        
        
        self.transform = transform
        self.crop_size = crop_size
        self.with_data_aug = with_data_aug
        self.labels_dict  = {}
        self.images_chunk = []
        self.images_names = []

        for l in self.labellist:
            label_name = l.split('.')[0]
            self.labels_dict[label_name] = self.im_loader(self.dataroot+'clear_images/'+l)
        
        for i, l in enumerate(self.imlist):
            im_name = l.split('_')[0]
            self.images_names.append(im_name)
            self.images_chunk.append(self.im_loader(self.dataroot+'hazy/'+l))

    # process data                                                                       
    def __getitem__(self, index):
        haze_img  = self.images_chunk[index]
        img_name  = self.images_names[index]
        label_img = self.labels_dict[img_name]
        if type(self.resize_to).__name__ == 'tuple':
            quit()
#            rainy_img = cv2.resize(rainy_img, self.resize_to)
#            label_img = cv2.resize(label_img, self.resize_to)
        else:
            pass
        
#        haze_img  = cv2.cvtColor(haze_img,  cv2.COLOR_BGR2RGB)
#        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

        if self.crop_size!= None:
            haze_img  = Image.fromarray(haze_img )
            label_img = Image.fromarray(label_img)

            W, H = label_img.size
            x_offset = random.randint(0, W - self.crop_size)
            y_offset = random.randint(0, H - self.crop_size)
            label_img = label_img.crop((x_offset, 
                                        y_offset, 
                                        x_offset+self.crop_size, 
                                        y_offset+self.crop_size))

            haze_img = haze_img.crop((x_offset, 
                                      y_offset, 
                                      x_offset+self.crop_size, 
                                      y_offset+self.crop_size))

        if self.with_data_aug:
            # Horizontal flip
            if random.random() > 0.5:                
                label_img = label_img.transpose(Image.FLIP_LEFT_RIGHT)
                haze_img  = haze_img.transpose( Image.FLIP_LEFT_RIGHT)

            # Vertical flip
#            if random.random() > 0.5:
#                label_img = label_img.transpose(Image.FLIP_TOP_BOTTOM)
#                blur_img = blur_img.transpose(Image.FLIP_TOP_BOTTOM)
            # Random rotation (90 or 180 or 270)
#            if random.random() > 0.5:
#                angle = random.choice([90, 180, 270])
#                label_img = label_img.rotate(angle)
#                blur_img = blur_img.rotate(angle)              

        label_img = np.asarray(label_img)
        haze_img  = np.asarray(haze_img )
        
        haze_img_M = cv2.resize(haze_img, 
                                (int(haze_img.shape[1]*0.5), 
                                 int(haze_img.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        haze_img_C = cv2.resize(haze_img_M, 
                                (int(haze_img_M.shape[1]*0.5), 
                                 int(haze_img_M.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        label_img_M = cv2.resize(label_img, 
                                (int(label_img.shape[1]*0.5), 
                                 int(label_img.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        label_img_C = cv2.resize(label_img_M, 
                                (int(label_img_M.shape[1]*0.5), 
                                 int(label_img_M.shape[0]*0.5)),
                                interpolation=cv2.INTER_CUBIC)

        haze_img   = self.transform(haze_img)
        haze_img_M = self.transform(haze_img_M)
        haze_img_C = self.transform(haze_img_C)

        label_img   = self.transform(label_img)
        label_img_M = self.transform(label_img_M)
        label_img_C = self.transform(label_img_C)

        return (haze_img, haze_img_M, haze_img_C), (label_img, label_img_M, label_img_C)


    def __len__(self):
        return len(self.labellist)
        
        

