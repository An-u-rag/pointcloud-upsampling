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

from pointnetpp import PointNetPPEncoder

from utils_samplers import *


class PUnet(nn.Module):
    def __init__(self, npoint=1024, nlayers=4, radii=[0.05, 0.1, 0.2, 0.3], nsamples=[32, 32, 32, 32], is_color=True, is_normal=False):
        super(PUnet, self).__init__()

        # Feature extraction with set abstraction layers (PointNet++)
        self.pointnetencoder = PointNetPPEncoder(
            npoint, nlayers, radii, nsamples, is_color, is_normal)

        # Feature Aggregation (PointNet++)

    def forward(self, xyz):
        l_xyz, l_points = self.pointnetencoder(xyz)
        return l_xyz, l_points


if __name__ == '__main__':
    import torch
    import time
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    model = PointNetPPEncoder(is_color=True, is_normal=True).to(device)
    xyz = torch.rand(6, 9, 1024).to(device)
    (model(xyz))
    print(f"Time Elapsed: {time.time() - start}")
