import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pietorch.N_modules as n_mods
from PIL import Image
from .utils import pixel_unshuffle
from .N_modules import FeatNorm, ConvLayer, SETVBasicBlock, UpSampleModule, SEBasicBlock, DownSampleModule

class DuRB_tv(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, k1_size=3, k2_size=3,
                 dilation=1, with_relu=True, up_type='PixShuf', relu_type='relu'):
        super(DuRB_tv, self).__init__()

        self.base = SETVBasicBlock(in_dim, in_dim, norm_type=None)

        # In T^{l}_{1}: A up-sample module + convolutional layer
        self.upsamp_module = UpSampleModule(in_dim, norm_type=None,
                                            module_type=up_type, relu_type=relu_type)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        
        # In T^{l}_{2}: A convolutional layer with stride = 2 for down-sampling
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=3, stride=2)
        
        self.down_se = SEBasicBlock(res_dim, res_dim, norm_type=None)
        self.down_seconv = ConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=2)
        self.down_atten = nn.Sequential(*[self.down_se, self.down_seconv])
        
        self.merge = ConvLayer(2*out_dim, out_dim, kernel_size=3, stride=1)
        self.with_relu = with_relu

        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU()
        else:
            print('what relu type?')
            
    def forward(self, x, res):
        
        # ---- The first two convs -----
        x_r = x
        x = self.base(x)
        # ------------------------------
        
        # T^{l}_{1}:
        x = self.upsamp_module(x)
        x = self.up_conv(x)
        
        x+= res
        x = self.relu(x)
        res = x
        
        # T^{l}_{2}:
        x1 = self.relu(self.down_conv(x))
        x2 = self.relu(self.down_atten(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.merge(x)
        x+= x_r
        x = self.relu(x)

        return x, res        
