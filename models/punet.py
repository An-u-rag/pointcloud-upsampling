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
# from models.sinkhorn import sinkhorn
from pytorch3d.loss import chamfer
# import pyemd


def knn(pc, n_neighbors=32):
    dist = torch.cdist(pc, pc)
    neigbhors = dist.topk(k=n_neighbors, dim=2, largest=False)
    return neigbhors.indices


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


# PUNet Loss Function
class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    def get_emd_loss(self, gp, gtp, pcd_radius):
        # gp_np = gp.cpu().numpy()
        # gtp_np = gtp.cpu().numpy()
        # emd_loss = torch.tensor([pyemd.emd(gp_np[i], gtp_np[i])
        #                         for i in range(gp_np.shape[0])])
        # emd_loss = [sinkhorn(gp[i], gtp[i]) for i in range(gp.shape[0])]
        # return torch.mean(emd_loss)/pcd_radius
        return 1

    def get_cd_loss(self, gp, gtp, pcd_radius):
        cd_loss, _ = chamfer.chamfer_distance(gp, gtp)
        return cd_loss

    def get_repulsion_loss(self, pred):
        # Get K nearest neighbors' indices of each point
        idx = knn(pred, self.nn_size)
        idx = idx[:, :, 1:]  # remove first one
        idx = idx.contiguous()  # B, N, nn

        pred = pred.contiguous()  # B, N, 3
        # (B, N, 3), (B, N, nn) => (B, N, nn, 3)
        grouped_points = index_points(pred, idx)

        pred = pred.transpose(1, 2)  # B, 3, N
        grouped_points = grouped_points.permute(0, 3, 1, 2)
        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(
            self.eps).to(grouped_points.device))
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        return uniform_loss

    def forward(self, pred, gt, pcd_radius):
        return self.get_cd_loss(pred, gt, pcd_radius) * 100, \
            self.alpha * self.get_repulsion_loss(pred)
        # return self.get_emd_loss(pred, gt, pcd_radius) * 100, \
        #     self.alpha * self.get_repulsion_loss(pred)


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
