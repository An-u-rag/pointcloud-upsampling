import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from models.pointnetpp import PointNetPPEncoder, PointNetPPDecoder

from models.utils_samplers import *


class PUnet(nn.Module):
    def __init__(self, npoint=1024, nlayers=4, radii=[0.05, 0.1, 0.2, 0.3], nsamples=[32, 32, 32, 32], is_color=True, is_normal=False):
        super(PUnet, self).__init__()
        self.mlplists = [
            [32, 32, 64],
            [64, 64, 128],
            [128, 128, 256],
            [256, 256, 512]
        ]
        # Feature extraction with set abstraction layers (PointNet++)
        self.pointnetencoder = PointNetPPEncoder(
            self.mlplists, npoint, nlayers, radii, nsamples, is_color, is_normal)

        # Feature Propagation and Interpolation (PointNet++)
        self.pointnetfp = PointNetPPDecoder(self.mlplists)

    def forward(self, xyz):
        l_xyz, l_points = self.pointnetencoder(xyz)

        out = self.pointnetfp(l_xyz, l_points)

        return out


if __name__ == '__main__':
    import torch
    import time
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = PUnet(is_color=True, is_normal=True).to(device)
    xyz = torch.rand(6, 9, 1024).to(device)
    (model(xyz))
    print(f"Time Elapsed: {time.time() - start}")
