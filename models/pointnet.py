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


# T-NET module that is used in PointNet for
# alignment of pointcloud to canonical space
# invariant to translation, rotation and scaling
class TNet(nn.Module):
    def __init__(self, channels, feature_transform=False):
        super(TNet, self).__init__()
        self.k = channels
        self.conv1 = nn.Conv1d(channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channels*channels)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # input and output is B x N x 3 ?
        B, C, N = x.size()
        print(B, C, N)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # Max of the features to form a global feature vector
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # Flatten for linear layers

        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # initialize transform matrix as an identity matrix and learn from that
        iden = torch.eye(self.k, requires_grad=True).view(
            1, self.k * self.k).repeat(B, 1)
        print(iden.shape)
        print(x.shape)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# PointNet Encoder class for the purpose of feature extraction from input point cloud
class PointNetEncoder(nn.Module):
    # Init function constructs the model with required parameters - eg: channels
    def __init__(self, in_channels=3, global_feat=True, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.stn = TNet(3)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TNet(64, True)

    def forward(self, x):  # input should be Batch x Num points x 3 (xyz)
        B, C, N = x.size()
        print(B, C, N)
        # Here if we have input with more than just x, y, z, then we need to:
        # 1. split to x and feature vectos
        # 2. matrix multiply x and transform
        # 3. concatenate feature to x along last dimension again to get x
        if C > 3:
            features = x[:, 3:, :]
            x = x[:, :3, :]
        T = self.stn(x)

        x = x.transpose(2, 1)
        x = torch.bmm(x, T)

        if C > 3:
            features = features.transpose(2, 1)
            x = torch.cat([x, features], dim=2)
        x = x.transpose(2, 1)  # [B, C, N]
        x = self.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            T_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, T_feat)
            x = x.transpose(2, 1)
        else:
            T_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:  # Only global features
            return x, T, T_feat
        else:  # Global + local features
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            print(x.size())
            input("X size after reshape and repeat")
            # Residual connection from pointfeat
            return torch.cat([x, pointfeat], 1), T, T_feat


if __name__ == '__main__':
    import torch
    model = PointNetEncoder(6)
    xyz = torch.rand(6, 6, 2048)
    (model(xyz))
