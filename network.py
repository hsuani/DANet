import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics import calc_GMetric
from torch.nn import init
from torch import optim


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_uniform_(m.weight.data, a=0.01, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def calc_mean_std(features):
    """
    channel wise mean and std
    :param features: shape of features -> [b, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[b, c, 1, 1]
    """
    b, c = features.size()[:2]
    features_mean = torch.mean(features.reshape(b, c, -1), dim=2, keepdim=True).reshape(b, c, 1, 1)
    features_std = torch.mean(features.reshape(b, c, -1), dim=2, keepdim=True).reshape(b, c, 1, 1) + 1e-6
    return features_mean, features_std

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x.float())
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
              nn.Upsample(scale_factor=2),
              nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
	      nn.BatchNorm2d(ch_out),
	      nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi


def iw_norm(feature, device):
    n, c, h, w = feature.size()
    diag = torch.eye(c, c).to(device)
    content_mean, content_std = calc_mean_std(feature)
    shift = feature - content_mean
    shift = shift.view(n, c, -1)
    c_feature = torch.bmm(shift, shift.transpose(2, 1)) / (h * w)
    return torch.pow((diag * c_feature).sum(dim=2, keepdim=True), -0.5) * shift  ## Xs


class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.encoder = encoder(img_ch=img_ch)
        self.decoder = decoder(output_ch=output_ch)
        print('AttU_Net initialized.')

    def forward(self, xs, gt, state='training'):
        xs, _ = xs
        d1_list = []
               
        if len(xs.size()) == 5:
            b, n, c, h, w = xs.size()
            x = xs.reshape(b * n, c, h, w)
            re_size = (b, n, 1, h, w)
        else:
            b, c, h, w = xs.size()
            x = xs
            re_size = (b, 1, h, w)
        
        x1, x2, x3, x4, x5 = self.encoder(x)
        d1 = self.decoder([x1, x2, x3, x4, x5])
        return d1.reshape(re_size), gt, {'embedding': x5}


class encoder(nn.Module):
    def __init__(self, img_ch):
        super(encoder, self).__init__()
        self.encoder = nn.ModuleList([
            conv_block(ch_in=img_ch, ch_out=64),
            conv_block(ch_in=64, ch_out=128),
            conv_block(ch_in=128, ch_out=256),
            conv_block(ch_in=256, ch_out=512),
            conv_block(ch_in=512, ch_out=1024)
        ])
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    def forward(self, x):
        xlist = []
        for m in self.encoder:
            x1 = m(x)
            xlist.append(x1)
            if len(xlist) < 5:
                x = self.maxpool(x1)
        return xlist

class decoder(nn.Module):
    def __init__(self, output_ch):
        super(decoder, self).__init__()
        self.decoder = nn.ModuleList([
            up_conv(ch_in=1024,ch_out=512),
            attention_block(F_g=512,F_l=512,F_int=256),
            conv_block(ch_in=1024, ch_out=512),
            up_conv(ch_in=512,ch_out=256),
            attention_block(F_g=256,F_l=256,F_int=128),
            conv_block(ch_in=512, ch_out=256),
            up_conv(ch_in=256,ch_out=128),
            attention_block(F_g=128,F_l=128,F_int=64),
            conv_block(ch_in=256, ch_out=128),
            up_conv(ch_in=128,ch_out=64),
            attention_block(F_g=64,F_l=64,F_int=32),
            conv_block(ch_in=128, ch_out=64),
            nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(output_ch)
        ])

    def forward(self, xs):
        s = -1
        d5, x4 = xs[s], xs[s-1]
        for i in range(4):
            d5 = self.decoder[i*3](d5)
            x4 = self.decoder[i*3+1](g=d5, x=x4)
            d5 = torch.cat((x4, d5), dim=1)
            d5 = self.decoder[i*3+2](d5)
            if i < 3:
                s = s - 1
                x4 = xs[s-1]
        d1 = self.decoder[-2](d5)
        d1 = self.decoder[-1](d1)
        return d1

class DamageNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(DamageNet, self).__init__()
        self.encoder = encoder(img_ch=img_ch)
        self.diff_decoder = decoder(output_ch=4)
        self.differ_att = conv_block(ch_in=1024*2, ch_out=1024)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, xs, gt, state='training'):
        xs_c, xs_t = xs
        d1_list = []
        result = {}

        x_c, self.re_size_c = self.input_resize(xs_c)
        x_t, self.re_size_t = self.input_resize(xs_t)

        featmap_c = self.encoder(x_c)
        featmap_t = self.encoder(x_t)
        featmap_cat = torch.cat((featmap_c[-1], featmap_t[-1]), 1)
        featmap_diff = self.differ_att(featmap_cat)
        featmap_t.append(featmap_diff)
        diff_xlist = [featmap_t[i] for i in range(4)]
        diff_xlist.append(featmap_t[-1])
        d1_diff = self.diff_decoder(diff_xlist)
        result.update({'d1_diff': d1_diff})
        return d1_diff.reshape(self.re_size_t), gt, result
        
    def input_resize(self, xs): 
        if len(xs.size()) == 5:
            b, n, c, h, w = xs.size()
            x = xs.reshape(b * n, c, h, w)
            re_size = (b, n, 4, h, w)  ## 4 type
        else:
            b, c, h, w = xs.size()
            x = xs
            re_size = (b, 4, h, w)
        return x, re_size

    def normalize(self, x, size, state):
        return x, torch.zeros(1).to(self.device)
        

