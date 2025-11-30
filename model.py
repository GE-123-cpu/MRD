import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class MultiProjectionLayer(nn.Module):
    def __init__(self, base=64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
    def forward(self, features):
        fa = self.proj_a(features[0])
        fb = self.proj_b(features[1])
        fc = self.proj_c(features[2])
        return [fa, fb, fc]


class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''

    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.LeakyReLU()
        #self.relu = torch.nn.ReLU()
        self.norm1 = nn.InstanceNorm2d(in_c // 2)
        self.norm2 = nn.InstanceNorm2d(in_c // 4)
        self.norm3 = nn.InstanceNorm2d(in_c // 2)
        self.norm4 = nn.InstanceNorm2d(in_c)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu(x)

        return x


class fuse_block_2(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c):
        super(fuse_block_2, self).__init__()
        self.conv1x1_downsample_dim = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=0.5, mode='bilinear'),
            torch.nn.Conv2d(in_c, 2 * in_c, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(2 * in_c),
            torch.nn.ReLU(inplace=True),
        )

        self.fuse = torch.nn.Sequential(
                torch.nn.Conv2d(2 * in_c, 2 * in_c, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(2 * in_c),
                torch.nn.ReLU(inplace=True)
            )

        self.catconv = torch.nn.Sequential(
                torch.nn.Conv2d(4 * in_c, 2 * in_c, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(2 * in_c),
                torch.nn.ReLU(inplace=True)
            )
        self.conv3x3_simattn = torch.nn.Sequential(
            torch.nn.Conv2d(2 * in_c, 2 * in_c, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(2 * in_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, d, p):
        with torch.no_grad():
             sim_a = torch.unsqueeze(F.cosine_similarity(d.detach(), p[2]), dim=1)
             sim_a_a = sim_a

        down = self.conv1x1_downsample_dim(p[1])
        fuse_f = self.fuse(down + p[2])
        fb = torch.cat([d, self.conv3x3_simattn(fuse_f * sim_a_a + d * (1 - sim_a_a))], dim=1)
        out = self.catconv(fb)
        return out