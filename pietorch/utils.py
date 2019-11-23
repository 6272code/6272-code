import random
import time
import datetime
import sys
from torch.autograd import Variable
import torch
from visdom import Visdom
import numpy as np
from PIL import Image
import cv2
from torch.nn import Parameter as Parameter

def tensor2image(image):
    image[image>1] = 1
    image[image<0] = 0
    image*= 255
    return image.astype(np.uint8)

class Logger():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}
        
        
    def log(self, losses=None, images=None):        

        if losses!= None:
            self.mean_period += (time.time() - self.prev_time)
            self.prev_time = time.time()
        
            sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

            for i, loss_name in enumerate(losses.keys()):
                if loss_name not in self.losses:
                    self.losses[loss_name] = losses[loss_name]
                else:
                    self.losses[loss_name] += losses[loss_name]
                    
                if (i+1) == len(losses.keys()):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name]/self.batch))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name]/self.batch))
    
            batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
            batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch

            # End of epoch
            if (self.batch % self.batches_epoch) == 0:
                # Plot losses
                for loss_name, loss in self.losses.items():
                    if loss_name not in self.loss_windows:
                        self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]), 
                                                                     Y=np.array([loss/self.batch]),
                                                                     opts={'xlabel': 'epochs', 
                                                                           'ylabel': loss_name, 
                                                                           'title': loss_name})
                    else:
                        self.viz.line(X=np.array([self.epoch]), Y=np.array([loss/self.batch]), win=self.loss_windows[loss_name], update='append')
                    # Reset losses for next k epoches
                    self.losses[loss_name] = 0.0
                    
                self.epoch += 1
                self.batch = 1
                sys.stdout.write('\n')
            else:
                self.batch += 1

            
        if images!= None:
            # Draw images
            for image_name, image_ in images.items():
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(tensor2image(image_), opts={'title':image_name})
                else:
                    self.viz.image(tensor2image(image_), win=self.image_windows[image_name], opts={'title':image_name})


def SaltAndPepper(img, probs):
    prob = random.choice(probs)
    c, h, w = img.size()
    rnd = torch.zeros((h, w)).normal_(0, std=1)
    minV = torch.min(rnd)
    maxV = torch.max(rnd)
    rnd = (rnd - minV)/(maxV - minV)
    rnd = rnd.expand(3, h ,w)

    pepper = (rnd < prob/2)
    salt = (rnd > 1 - prob/2)

    img[pepper] = 0.
    img[salt] = 1.
        
    return img

def GaussianNoise(img, Vars):
    c, h, w = img.size()
    var = random.choice(Vars)
    if var!= 0:
        noise_pad = torch.FloatTensor(c, h, w).normal_(0, var)
        noise_pad = torch.div(noise_pad, 255.0)
        img+= noise_pad

    else:
        pass
        
    return img


def align_to_k(img, k):
    a_row = int(img.shape[0]/k)*k
    a_col = int(img.shape[1]/k)*k
    img = img[0:a_row, 0:a_col]
    return img


def GaussianBlur(img, Vars):    
    var = random.choice(Vars)
    if var!= 0:
        img = np.asarray(img)
        img = cv2.GaussianBlur(img, (21, 21), var)
        img = Image.fromarray(img)
    else:
        pass
        
    return img


def load_part_of_pretrained(current_model, pre_model):

#    pre_dict = {k: v for k, v in pre_model.items() if k in current_dict}
    
    current_dict = current_model.state_dict()        
    loaded_dict = {k: pre_model[k] for k in current_dict.keys() if k in pre_model.keys()}
    for name, param in loaded_dict.items():
        if isinstance(param, Parameter):
            param = param.data
            
        try:
            current_dict[name].copy_(param)
        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the current model are {} and '
                               'whose dimensions in the pretrained model are {}.'
                               .format(name, current_dict[name].size(), param.size()))
    

def pixel_unshuffle(fm, r):
    b, c, h, w = fm.size()
    out_channel = c*(r**2)
    out_h = h//r
    out_w = w//r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)
    return fm_prime

def activation_func(relu_type):
    if relu_type == 'relu':
        return nn.ReLU()
    elif relu_type == 'prelu':
        return nn.PReLU()
