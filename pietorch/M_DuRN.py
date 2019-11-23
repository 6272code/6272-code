import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from .N_modules import FeatNorm, ConvLayer, SEBasicBlock, UpSampleModule, SELayer, C2FShareHead, NormalDecoderUp1, NormalDecoderUp2, LSTMHeadDown1, NormalHeadDown2, C2FShareTail
from .DuRBs import DuRB_tv

class EnDNet(nn.Module):
    def __init__(self, img_dim, norm_type=None, rain_iter=4):
        super(EnDNet, self).__init__()
        dim = 96
        res_dim = 64
        self.iteration = rain_iter
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # --- Input branches ---
        self.rain_head = LSTMHeadDown1()
        self.blur_head = C2FShareHead(need_res2=False)
        self.jpeg_head = C2FShareHead(need_res2=False)
        self.haze_head = NormalHeadDown2()
        
        # --- stem net ---
        self.DuRB1 = DuRB_tv(dim, dim, res_dim, k1_size=3, k2_size=3, dilation=2) # DuRB_tv means DuRB_M
        self.DuRB2 = DuRB_tv(dim, dim, res_dim, k1_size=5, k2_size=3, dilation=1)
        self.DuRB3 = DuRB_tv(dim, dim, res_dim, k1_size=3, k2_size=5, dilation=2)
        self.DuRB4 = DuRB_tv(dim, dim, res_dim, k1_size=5, k2_size=5, dilation=1)
        self.DuRB5 = DuRB_tv(dim, dim, res_dim, k1_size=7, k2_size=5, dilation=1)

        # --- Output branch1 = g1
        self.dec_1 = NormalDecoderUp1() # Rain
        self.DuRB6 = DuRB_tv(dim, dim, res_dim, k1_size=7, k2_size=5, dilation=2)

        # --- Output branch2 = DuRB6 + g2
        self.dec_2 = C2FShareTail(sum_before=False)  # Blur
        self.DuRB7 = DuRB_tv(dim, dim, res_dim, k1_size=11, k2_size=5, dilation=1)

        # --- Output branch3 = DuRB6 + DuRB7 + g3
        self.dec_3 = C2FShareTail(sum_before=False)  # Jepg
        self.DuRB8 = DuRB_tv(dim, dim, res_dim, k1_size=11, k2_size=5, dilation=1)

        # --- Output branch3 = DuRB6 + DuRB7 + DuRB8 + g4
        self.dec_4 = NormalDecoderUp2() # Haze 
    
    def forward(self, coarse, mid, fine, name):
        # --- Header ---
        if name  == 'rain':  # H, W == 128, 128.
            x = fine         # mid, coarse are not used.
            b,_,H,W = x.size()
            hidden  = Variable(torch.zeros(b, 32, H, W)).cuda()
            cell    = Variable(torch.zeros(b, 32, H, W)).cuda()
            stats   = [hidden, cell]
            input_img = x
            for i in range(self.iteration):
                x, res, stats = self.rain_head(x, input_img, stats)
                x, res = self.DuRB1(x, res)
                x, res = self.DuRB2(x, res)
                x, res = self.DuRB3(x, res)
                x, res = self.DuRB4(x, res)
                x, res = self.DuRB5(x, res)
                x      = self.dec_1(x, input_img)
            return x

        elif name == 'blur':
            x = coarse
            x, res = self.blur_head(x, scale='coarse')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            coarse_out = self.dec_2(x, coarse, scale='coarse')
            coarse_up  = F.interpolate(coarse_out, scale_factor=2, 
                                       mode='bicubic')
            x = mid
            x = torch.cat((x, coarse_up), dim=1)
            x, res = self.blur_head(x, scale='mid')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            mid_out = self.dec_2(x, mid, scale='mid')
            mid_up  = F.interpolate(mid_out, scale_factor=2, mode='bicubic')

            x = fine
            x = torch.cat((x, mid_up), dim=1)
            x, res = self.blur_head(x, scale='fine')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            fine_out = self.dec_2(x, fine, scale='fine')
            return coarse_out, mid_out, fine_out

        elif name == 'jpeg':
            x = coarse
            x, res = self.jpeg_head(x, scale='coarse')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            x, res = self.DuRB7(x, res)
            coarse_out = self.dec_3(x, coarse, scale='coarse')
            coarse_up  = F.interpolate(coarse_out, scale_factor=2, 
                                       mode='bicubic')
            x = mid
            x = torch.cat((x, coarse_up), dim=1)
            x, res = self.jpeg_head(x, scale='mid')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            x, res = self.DuRB7(x, res)
            mid_out = self.dec_3(x, mid, scale='mid')
            mid_up  = F.interpolate(mid_out, scale_factor=2, mode='bicubic')

            x = fine
            x = torch.cat((x, mid_up), dim=1)
            x, res = self.jpeg_head(x, scale='fine')
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            x, res = self.DuRB7(x, res)
            fine_out = self.dec_3(x, fine, scale='fine')
            return coarse_out, mid_out, fine_out

        else: # haze
            x = fine
            input_img = x
            x, res = self.haze_head(x)
            x, res = self.DuRB1(x, res)
            x, res = self.DuRB2(x, res)
            x, res = self.DuRB3(x, res)
            x, res = self.DuRB4(x, res)
            x, res = self.DuRB5(x, res)
            x, res = self.DuRB6(x, res)
            x, res = self.DuRB7(x, res)
            x, res = self.DuRB8(x, res)
            x = self.dec_4(x, input_img)
            return x
