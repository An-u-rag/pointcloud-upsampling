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
from pytorch3d.loss import chamfer


class GaussianDiffuser(nn.Module):
    def __init__(self, r=4):
        super().__init__()

        self.r = r

    def forward(self, pc):
        pass


class GaussianDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, pc):
        pass


if __name__ == '__main__':
    import torch
    import time
    start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # First use Gaussian Diffuser to get noisy samples at each layer
    pc = torch.rand(6, 9, 1024).to(device)

    enc = GaussianDiffuser(r=4)
    pc_noisy_layers = enc(pc)
    model = GaussianDecoder()
    (model(pc))
