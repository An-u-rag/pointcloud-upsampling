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


class FExpMLP(nn.Module):
    def __init__(self, mlplist):
        super(FExpMLP, self).__init__()
        self.conv_mlp = nn.ModuleList()
        for i in range(len(mlplist)-1):
            self.conv_mlp.append(nn.Conv2d(mlplist[i], mlplist[i+1], 1))

    def forward(self, x):
        for i in range(len(self.conv_mlp)):
            x = F.relu(self.conv_mlp[i](x))

        return x


class PUnet(nn.Module):
    def __init__(self, npoint=1024, nlayers=4, radii=[0.05, 0.1, 0.2, 0.3], nsamples=[32, 32, 32, 32], is_color=True, is_normal=False):
        super(PUnet, self).__init__()
        self.nlayers = nlayers
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

        # Feature Expansion Module (Basically an MLP to take 64 -> 256 -> 128)
        # since all n layers with 64 features are concatted + xyz input
        FExp_inputs = [(self.nlayers * 64) + 3, 256, 128]
        self.punetFExp = nn.ModuleList()
        for i in range(self.nlayers):  # Each layer goes through a separate conv mlp block
            self.punetFExp.append(FExpMLP(FExp_inputs))

        # Coordinate Regression and Reconstruction layers (input is # [B, 128, 4 * 1024, 1])
        in_ch = 128
        self.punetCR = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, xyz):
        # Send input point cloud to pointnet++ for feature extraction and get layerwise features
        l_xyz, l_points = self.pointnetencoder(xyz)
        # Send layer wise features for feature propagation and interpolation
        # Each layer has N points now with 64 features each
        # So that each layer features are B x 64 x N
        up_feats = self.pointnetfp(l_xyz, l_points)

        # Aggregate the features in up_feats (Concat along channel dimension)

        feats = torch.cat([xyz[:, :3, :].contiguous(),  # [B, 3, 1024]
                           l_points[1],  # [B, 64, 1024]
                           # ([B, 64, 1024], [B, 64, 1024], [B, 64, 1024])
                           *up_feats
                           ], dim=1).unsqueeze(-1)  # [B, 259, 1024, 1]

        # Now we expand the features using feature expansion module
        r_feats = []
        for k in range(len(self.punetFExp)):
            # bs, mid_ch, N, 1 # [B, 128, 1024, 1]
            feat_k = self.punetFExp[k](feats)
            r_feats.append(feat_k)
        # bs, mid_ch, r * N, 1 # [B, 128, 4 * 1024, 1]
        r_feats = torch.cat(r_feats, dim=2)

        # Coordinate regression and Reconstruction
        # Send the expanded point features through a mlp block for coordinate regression
        out = self.punetCR(r_feats)
        return out.squeeze(-1).transpose(1, 2).contiguous()


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
