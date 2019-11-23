import numpy as np
import torch
import cv2
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
from .utils import pixel_unshuffle, activation_func

# Convolutional layer
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, 
                 stride=1, dilation=1, pad_type='reflect', groups=1, bias=True):        
        super(ConvLayer, self).__init__()               
        # pad size
        padding = int(dilation * (kernel_size - 1) / 2)

        # pading type
        self.pad_type = pad_type        
        if pad_type == 'reflect':
            self.reflection_pad = nn.ReflectionPad2d(padding)
            self.conv2d         = nn.Conv2d(int(in_dim), int(out_dim), kernel_size, stride,
                                            dilation=dilation, groups=groups, bias=bias)            
        elif pad_type == 'zero':
            self.conv2d = nn.Conv2d(int(in_dim), int(out_dim), kernel_size, stride,
                                    padding=padding, dilation=dilation, groups=groups, bias=bias)
        else:
            raise Exception('pad_type is not found.')
            
    def forward(self, x):
        if self.pad_type == 'reflect':
            out = self.reflection_pad(x)
            out = self.conv2d(out)            
        else:
            out = self.conv2d(x)
        return out


class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if   norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        elif norm_type == "switchnorm2d":
            self.norm = SwitchNorm2d(dim)
        elif norm_type == "group":
            self.norm = GroupNorm(dim, num_groups=1)
        elif norm_type == "layernorm":
            self.norm = LayerNorm(dim)
        else:
            raise Exception("Normalization type not found.")
    def forward(self, x):
        return self.norm(x)


# The basic SE-ResNet module
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, reduction=96, norm_type=None):
        super(SEBasicBlock, self).__init__()        
        self.norm_type = norm_type
        
        if not norm_type is None:
            self.norm1 = FeatNorm(norm_type, planes)
            self.norm2 = FeatNorm(norm_type, planes)
        else:
            pass            
        self.conv1 = ConvLayer(inplanes, planes, 3, stride=1, pad_type='zero', bias=False)
        self.conv2 = ConvLayer(  planes, planes, 3, stride=1, pad_type='zero', bias=False)        
        self.se    = SELayer(    planes, reduction)
        self.relu  = nn.ReLU(inplace=True)        

    def forward(self, x):
        out = self.conv1(x)        
        if not self.norm_type is None:
            out = self.norm1(out)            
        out = self.relu(out)

        out = self.conv2(out)        
        if not self.norm_type is None:        
            out = self.norm2(out)
            
        out = self.se(out)
        out = out + x
        out = self.relu(out)
        return out


# The basic SETV-ResNet module
class SETVBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, reduction=64, norm_type=None, tv_beta=3): 
        super(SETVBasicBlock, self).__init__()        
        self.norm_type = norm_type
        
        if not norm_type is None:
            self.norm1 = FeatNorm(norm_type, planes)
            self.norm2 = FeatNorm(norm_type, planes)
        else:
            pass            
        self.conv1 = ConvLayer(inplanes, planes, 3, stride=1, pad_type='zero', bias=False)
        self.conv2 = ConvLayer(  planes, planes, 3, stride=1, pad_type='zero', bias=False)        
        self.se = SETVLayer(     planes, reduction, tv_beta=tv_beta)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, x):
        out = self.conv1(x)        
        if not self.norm_type is None:
            out = self.norm1(out)            
        out = self.relu(out)

        out = self.conv2(out)        
        if not self.norm_type is None:        
            out = self.norm2(out)
            
        out = self.se(out)
        out = out + x
        out = self.relu(out)
        return out


# For ablation study
class TVBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, reduction=64, norm_type=None): 
        super(TVBasicBlock, self).__init__()        
        self.norm_type = norm_type
        
        if not norm_type is None:
            self.norm1 = FeatNorm(norm_type, planes)
            self.norm2 = FeatNorm(norm_type, planes)

        self.conv1 = ConvLayer(inplanes, planes, 3, stride=1, pad_type='zero', bias=False)
        self.conv2 = ConvLayer(planes, planes, 3, stride=1, pad_type='zero', bias=False)        
        self.se = TVLayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, x):
        out = self.conv1(x)        
        if not self.norm_type is None:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)        
        if not self.norm_type is None:  
            out = self.norm2(out)
            
        out = self.se(out)
        out = out + x
        out = self.relu(out)
        return out
                      

# SE layer 
class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y        


# SE_tv layer 
class SETVLayer(nn.Module):
    def __init__(self, channel, reduction, tv_beta=3):
        super(SETVLayer, self).__init__()
        self.tv_beta = tv_beta
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(2*channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y_rv = (torch.abs(x[:,:,:-1, :] - x[:,:,1:, :]).pow(self.tv_beta)).view(b,c,-1).mean(dim=2)
        y_cv = (torch.abs(x[:,:,:, :-1] - x[:,:,:, 1:]).pow(self.tv_beta)).view(b,c,-1).mean(dim=2)
        y_tv = y_rv + y_cv
        y_tv = y_tv.view(b, c)
        y    = torch.cat((y, y_tv), dim=1)
        y    = self.fc(y).view(b, c, 1, 1)
        return x * y        


# No GAP
class TVLayer(nn.Module):
    def __init__(self, channel, reduction, tv_beta=3):
        super(TVLayer, self).__init__()
        self.tv_beta = tv_beta
        self.fc = nn.Sequential(
                nn.Linear(channel, reduction),
                nn.ReLU(inplace=True),
        nn.Linear(reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y_rv = (torch.abs(x[:,:,:-1, :] - x[:,:,1:, :]).pow(self.tv_beta)).view(b,c, -1).mean(dim=2)
        y_cv = (torch.abs(x[:,:,:, :-1] - x[:,:,:, 1:]).pow(self.tv_beta)).view(b,c, -1 ).mean(dim=2)
        y_tv = y_rv + y_cv
        y_tv = y_tv.view(b, c)
        y_tv = self.fc(y_tv).view(b, c, 1, 1)
        return x * y_tv
        
        
class UpSampleModule(nn.Module):
    def __init__(self, dim, norm_type, module_type, relu_type='relu',
                 kernel_size=1, ps_scale=4, TransConv_out=128):
        super(UpSampleModule, self).__init__()
        self.mod_type = module_type

        if self.mod_type == 'PixShuf':
            if not norm_type is None:
                self.upsamp = nn.Sequential(            
                    ConvLayer(dim, ps_scale*dim, kernel_size, 1),
                    FeatNorm(norm_type, ps_scale*dim),
                    nn.PixelShuffle(2))
            else:
                self.upsamp = nn.Sequential(
                    ConvLayer(dim, ps_scale*dim, kernel_size, 1),
                    nn.PixelShuffle(2))
        elif self.mod_type == 'TransConv':
            if relu_type == 'relu':
                self.upsamp = nn.Sequential(
                    nn.ConvTranspose2d(dim, TransConv_out, 4, 2, 1),
                    nn.ReflectionPad2d((1, 0, 1, 0)),
                    nn.AvgPool2d(2, stride = 1),
                    nn.ReLU())
            elif relu_type == 'prelu':
                self.upsamp = nn.Sequential(
                    nn.ConvTranspose2d(dim, TransConv_out, 4, 2, 1),
                    nn.ReflectionPad2d((1, 0, 1, 0)),
                    nn.AvgPool2d(2, stride = 1),
                    nn.PReLU())
            else:
                print('Unknown actiavation type for TransConv')
        else:
            raise Exception('Unknown up-sampling module type')

    def forward(self, x):
        return self.upsamp(x)


class DownSampleModule(nn.Module):
    def __init__(self, norm_type, module_type):
        super(DownSampleModule, self).__init__()

        self.mod_type = module_type
        if self.mod_type == 'PixUnShuf':
            pass
        else:
            raise Exception('Unknown down-sampling module type.')

    def forward(self, x):
        if self.mod_type == 'PixUnShuf':
            return pixel_unshuffle(x, 2)
        else:
            print(self.mod_type+'is not implemented for down-sampling.')
            quit()



# instance normalization implementation
class InsNorm(nn.Module):    
    def __init__(self, dim, eps=1e-9):
        super(InsNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps   = eps
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):        
        flat_len = x.size(2)*x.size(3)
        vec = x.view(x.size(0), x.size(1), flat_len)
        mean = torch.mean(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        var = torch.var(vec, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((flat_len - 1)/float(flat_len))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var+self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
       

class Coarse2FineHead(nn.Module):
    def __init__(self, in_dim=3, feat_dim=64):
        super(Coarse2FineHead, self).__init__()
        self.F_conv1 = ConvLayer(      in_dim, feat_dim/2, kernel_size=3)
        self.F_conv2 = ConvLayer(  feat_dim/2, feat_dim,   kernel_size=3)
        self.F_conv3 = ConvLayer(  feat_dim*4, feat_dim,   kernel_size=3)

        self.M_conv1 = ConvLayer(    in_dim, feat_dim, kernel_size=3)
        self.M_conv2 = ConvLayer(feat_dim*2, feat_dim, kernel_size=3)
        self.M_conv3 = ConvLayer(feat_dim*4, feat_dim, kernel_size=3)

        self.C_conv1 = ConvLayer(    in_dim, feat_dim, kernel_size=3)
        self.C_conv2 = ConvLayer(feat_dim*2, feat_dim, kernel_size=3)
        self.C_conv3 = ConvLayer(feat_dim*4, feat_dim, kernel_size=3)
        
        self.F_downsamp = DownSampleModule(feat_dim, None, 'PixUnShuf')
        self.M_downsamp = DownSampleModule(feat_dim, None, 'PixUnShuf')
        self.C_downsamp = DownSampleModule(feat_dim, None, 'PixUnShuf')
        self.relu = nn.PReLU()

    def forward(self, F_x, M_x, C_x):
        F_img_residual  = F_x
        F_x = self.relu(self.F_conv1(F_x))
        F_x = self.relu(self.F_conv2(F_x))
        F_feat_residual = F_x
        F_x = self.F_downsamp(F_x)
        F_x = self.relu(self.F_conv3(F_x))
        F_output = [F_img_residual, F_feat_residual, F_x]

        M_img_residual  = M_x
        M_x = self.relu(self.M_conv1(M_x))
        M_x = torch.cat((M_x, F_x), dim=1)
        M_x = self.relu(self.M_conv2(M_x))
        M_feat_residual = M_x
        M_x = self.M_downsamp(M_x)
        M_x = self.relu(self.M_conv3(M_x))
        M_output = [M_img_residual, M_feat_residual, M_x]

        C_img_residual  = C_x
        C_x = self.relu(self.C_conv1(C_x))
        C_x = torch.cat((C_x, M_x), dim=1)
        C_x = self.relu(self.C_conv2(C_x))
        C_feat_residual = C_x
        C_x = self.C_downsamp(C_x)
        C_x = self.relu(self.C_conv3(C_x))
        C_output = [C_img_residual, C_feat_residual, C_x]

        return F_output, M_output, C_output

        
class Coarse2FineTail(nn.Module):
    def __init__(self, feat_dim=64, out_dim=3):
        super(Coarse2FineTail, self).__init__()
        self.C_upsamp = UpSampleModule(feat_dim, None, "PixShuf")
        self.M_upsamp = UpSampleModule(feat_dim, None, "PixShuf")
        self.F_upsamp = UpSampleModule(feat_dim, None, "PixShuf")

        self.C_conv1 = ConvLayer(feat_dim, feat_dim, kernel_size=3)
        self.C_conv2 = ConvLayer(feat_dim, feat_dim, kernel_size=3)
        self.C_conv3 = ConvLayer(feat_dim,  out_dim, kernel_size=3)

        self.M_conv1 = ConvLayer(feat_dim*2, feat_dim, kernel_size=3)
        self.M_conv2 = ConvLayer(feat_dim,   feat_dim, kernel_size=3)
        self.M_conv3 = ConvLayer(feat_dim,    out_dim, kernel_size=3)

        self.F_conv1 = ConvLayer(feat_dim*2, feat_dim, kernel_size=3)
        self.F_conv2 = ConvLayer(feat_dim,   feat_dim/2, kernel_size=3)
        self.F_conv3 = ConvLayer(feat_dim/2,    out_dim, kernel_size=3)
        self.relu = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, F_data, M_data, C_data):
        F_img_residual, F_x = F_data
        M_img_residual, M_x = M_data
        C_img_residual, C_x = C_data

        C_x = self.relu(self.C_conv1(C_x))
        C_x = self.C_upsamp(C_x)
        C_x = self.relu(self.C_conv2(C_x))
        C2M = C_x
        C_x = self.tanh(self.C_conv3(C_x))
        C_x = C_x + C_img_residual

        M_x = torch.cat((M_x, C2M), dim=1)
        M_x = self.relu(self.M_conv1(M_x))
        M_x = self.M_upsamp(M_x)
        M_x = self.relu(self.M_conv2(M_x))
        M2F = M_x
        M_x = self.tanh(self.M_conv3(M_x))
        M_x = M_x + M_img_residual

        F_x = torch.cat((F_x, M2F), dim=1)
        F_x = self.relu(self.F_conv1(F_x))
        F_x = self.F_upsamp(F_x)
        F_x = self.relu(self.F_conv2(F_x))
        F_x = self.tanh(self.F_conv3(F_x))
        F_x = F_x + F_img_residual
        
        return F_x, M_x, C_x
        

class C2FShareHead(nn.Module):
    def __init__(self, in_dim=3, need_res2=True):
        super(C2FShareHead, self).__init__()
        self.need_res2 = need_res2

        self.F_conv1 = ConvLayer(3+3, 32, kernel_size=3)
        self.M_conv1 = ConvLayer(3+3, 32, kernel_size=3)
        self.C_conv1 = ConvLayer(3, 32, kernel_size=3)

        self.conv1 = ConvLayer(32, 32, kernel_size=3)
        self.conv2 = ConvLayer(32, 32, kernel_size=3)

        self.F_conv2 = ConvLayer(32*4, 64, kernel_size=3)
        self.M_conv2 = ConvLayer(32*4, 64, kernel_size=3)
        self.C_conv2 = ConvLayer(32*4, 64, kernel_size=3)

        self.conv3  = ConvLayer(64, 64, kernel_size=3)
        self.conv4  = ConvLayer(64, 64, kernel_size=3)

        self.F_conv3 = ConvLayer(64*4, 96, kernel_size=3)
        self.M_conv3 = ConvLayer(64*4, 96, kernel_size=3)
        self.C_conv3 = ConvLayer(64*4, 96, kernel_size=3)
        
        self.downsamp = DownSampleModule(None, 'PixUnShuf')
        self.relu = nn.ReLU()

    def forward(self, x, scale):
        if scale   == 'fine':
            x = self.relu(self.F_conv1(x))
        elif scale == 'mid':
            x = self.relu(self.M_conv1(x))
        elif scale == 'coarse':
            x = self.relu(self.C_conv1(x))
        else:
           raise Exception('scale unknown.')

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        if self.need_res2:
            res2 = x

        x = self.downsamp(x)

        if scale   == 'fine':
            x = self.relu(self.F_conv2(x))
        elif scale == 'mid':
            x = self.relu(self.M_conv2(x))            
        elif scale == 'coarse':
            x = self.relu(self.C_conv2(x))
        else:
            raise Exception('scale unknown.')

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        res1 = x
        x = self.downsamp(x)

        if scale   == 'fine':
            x = self.relu(self.F_conv3(x))
        elif scale == 'mid':
            x = self.relu(self.M_conv3(x))            
        elif scale == 'coarse':
            x = self.relu(self.C_conv3(x))
        else:
            raise Exception('scale unknown.')

        if self.need_res2:
            return x, res1, res2
        else:
            return x, res1


class C2FShareTail(nn.Module):
    def __init__(self, sum_before=True):
        super(C2FShareTail, self).__init__()
        self.sum_before = sum_before

        self.C_upsamp1 = UpSampleModule(96, None, "PixShuf")
        self.C_upconv1 = ConvLayer(96, 64, kernel_size=3)
        self.M_upsamp1 = UpSampleModule(96, None, "PixShuf")
        self.M_upconv1 = ConvLayer(96, 64, kernel_size=3)
        self.F_upsamp1 = UpSampleModule(96, None, "PixShuf")
        self.F_upconv1 = ConvLayer(96, 64, kernel_size=3)
        self.conv1     = ConvLayer(64, 64, kernel_size=3)
        self.conv2     = ConvLayer(64, 64, kernel_size=3)

        self.C_upsamp2 = UpSampleModule(64, None, "PixShuf")
        self.C_upconv2 = ConvLayer(64, 32, kernel_size=3)
        self.M_upsamp2 = UpSampleModule(64, None, "PixShuf")
        self.M_upconv2 = ConvLayer(64, 32, kernel_size=3)
        self.F_upsamp2 = UpSampleModule(64, None, "PixShuf")
        self.F_upconv2 = ConvLayer(64, 32, kernel_size=3)
        self.conv3     = ConvLayer(32, 32, kernel_size=3)
        self.conv4     = ConvLayer(32, 32, kernel_size=3)

        self.C_conv = ConvLayer(32, 3, kernel_size=3)
        self.M_conv = ConvLayer(32, 3, kernel_size=3)
        self.F_conv = ConvLayer(32, 3, kernel_size=3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, res, scale):
        if scale == 'fine':
            x = self.F_upsamp1(x)
            x = self.relu(self.F_upconv1(x))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.F_upsamp2(x)
            x = self.relu(self.F_upconv2(x))
            x = self.relu(self.conv3(x))
            if self.sum_before:
                x = self.conv4(x)
                x = x+res
                x = self.relu(x)
                res = x
                x = self.F_conv(x)
            else:
                x = self.relu(x)
                x = self.tanh(self.C_conv(x))
                x = x + res

        elif scale == 'mid':
            x = self.M_upsamp1(x)
            x = self.relu(self.M_upconv1(x))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.M_upsamp2(x)
            x = self.relu(self.M_upconv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            if self.sum_before:
                x = x+res
                x = self.relu(x)
                res = x
                x = self.M_conv(x)
            else:
                x = self.relu(x)
                x = self.tanh(self.C_conv(x))
                x = x + res
                

        elif scale == 'coarse':
            x = self.C_upsamp1(x)
            x = self.relu(self.C_upconv1(x))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.C_upsamp2(x)
            x = self.relu(self.C_upconv2(x))
            x = self.relu(self.conv3(x))
            x = self.conv4(x)
            if self.sum_before:
                x = x+res
                x = self.relu(x)
                res = x
                x = self.C_conv(x)
            else:
                x = self.relu(x)
                x = self.tanh(self.C_conv(x))
                x = x + res
        else:
            print('fuck')

        if self.sum_before:
           if scale!= 'fine':
               return res, x
           else:
               return x
        else:
            return x


class single_convlstm(nn.Module):
    def __init__(self, dim=32):
        super(single_convlstm, self).__init__()
        self.comb_conv = ConvLayer(3+3, dim, 3)
        self.conv_i = nn.Sequential(ConvLayer(dim+dim, dim, 3),
                                    nn.Sigmoid())
        self.conv_f = nn.Sequential(ConvLayer(dim+dim, dim, 3),
                                    nn.Sigmoid())
        self.conv_g = nn.Sequential(ConvLayer(dim+dim, dim, 3),
                                    nn.Tanh())
        self.conv_o = nn.Sequential(ConvLayer(dim+dim, dim, 3),
                                    nn.Sigmoid())
        self.relu = nn.ReLU()
    def forward(self, x, stats):
        # hidden, cell are tensors, x is cat(img1, img2)
        hidden, cell = stats
        x = self.relu(self.comb_conv(x))
        
        x = torch.cat((x, hidden), dim=1)
        i_x  = self.conv_i(x)
        f_x  = self.conv_f(x)
        g_x  = self.conv_g(x)
        o_x  = self.conv_o(x)
        cell = f_x * cell + i_x * g_x
        hidden = o_x * torch.tanh(cell)
        x = hidden

        stats = [hidden, cell]
        return x, stats


class LSTMHeadDown1(nn.Module):
    def __init__(self, in_dim=3):
        super(LSTMHeadDown1, self).__init__()
        self.lstm  = single_convlstm()
        self.conv1 = ConvLayer(32,   64, kernel_size=3)
        self.conv2 = ConvLayer(64*4, 96, kernel_size=3)
        self.relu  = nn.ReLU()

    def forward(self, x, input_img, stats):
        x = torch.cat((x, input_img), dim=1)
        x, stats = self.lstm(x, stats)
        x = self.relu(self.conv1(x))
        res = x
        x = pixel_unshuffle(x, 2)
        x = self.relu(self.conv2(x))
        return x, res, stats

class NormalHeadDown2(nn.Module):
    def __init__(self, in_dim=3):
        super(NormalHeadDown2, self).__init__()
        self.conv1 = ConvLayer(in_dim, 32, kernel_size=3)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv3 = ConvLayer(64, 96, kernel_size=3, stride=2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        res = x
        x = self.relu(self.conv3(x))
        return x, res


class NormalDecoderUp2(nn.Module):
    def __init__(self):
        super(NormalDecoderUp2, self).__init__()
        self.up1   = UpSampleModule(dim=96, norm_type=None, module_type='PixShuf')
        self.conv1 = ConvLayer(96, 64, kernel_size=3, stride=1)
        self.up2   = UpSampleModule(dim=64, norm_type=None, module_type='PixShuf')
        self.conv2 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv3 = ConvLayer(32, 3,  kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, residual):
        x = self.relu(self.conv1(self.up1(x)))
        x = self.relu(self.conv2(self.up2(x)))
        x = self.tanh(self.conv3(x))
        x = x + residual
        return x

class NormalDecoderUp1(nn.Module):
    def __init__(self):
        super(NormalDecoderUp1, self).__init__()
        self.up1   = UpSampleModule(dim=96, norm_type=None, 
                                    module_type='PixShuf', ps_scale=2)
        self.conv1 = ConvLayer(48, 48, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(48, 3,  kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, residual):
        x = self.relu(self.conv1(self.up1(x)))
        x = self.tanh(self.conv2(x))
        x = x + residual
        return x
        
        
# ----------------------------------------------------

# Data augmentation Lambda
# x has to be in PIL.Image
def customized_image_rotation(x, p=0.5):
    if not isinstance(x, Image.Image):
        raise TypeError('An input image should be PIL.Image. Got {}'.format(type(x)))
        
    angles = [90, 180, 270]
    
    if p < random.random():
        return x

    else:
        angle = random.choice(angles)
        return x.rotate(angle)
            

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y        


class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False
        
    def cuda(self):
        self.gpu = True
        
    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return Variable(noise).cuda()
        else:
            return Variable(noise)
        
    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor
    
    def gumbel_softmax_sample(self, logits, temperature):
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples
    
    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()
        
        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)




class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=2):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)

        return out * x
                                            
