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

from models.pointnet import PointNetEncoder, TNet

from models.utils_samplers import *


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    # Aim of this module is to get downsampled point cloud with the downsampled features
    # It also downsamples to multiple scales and gets features at each of those levels
    # This is better than normal SetAbstraction since it is done at multiple scales or radii
    # helps extract local information better
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            # We add 3 here again since we will concatenate the normalized xyz positional coords to it
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        # We get the S sampled centroid indices from the farthest_point_sample() function
        # Then we pass points and the indices to the index points function to get the exact coordiantes of those indices
        # in [B, S, C] tensor format
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        # Now we do the sampling around each centroid using ball query
        # to sample nsample points (in nsample_list) around each centroid with radius in radius_list
        # (Hierarchical sampling in Multi Scale Sampling (MSG))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # Select current number of points to sampel aroudn each centroid as K
            K = self.nsample_list[i]
            # Use Ball query to sample those K points and return their indices
            # Input to ball query function is radius, no. of points to sample, all points, centroid points
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            # Convert those indices to xyz coordinates
            grouped_xyz = index_points(xyz, group_idx)
            # Normalize each local poitn cloud for each centroid so that they are invariant to rotation and translation
            # Here we subtract each sampled point from ball query from their respective centers
            #### (Don't know why new_xyz is beign reshaped before subtraction, gotta find out) ####
            # Possible because the new_xyz shape is [B, S, C] whereas grouped_xyz is [B, K, C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            # Similarly we get the whole points (not just point location) with features based on indices
            if points is not None:
                grouped_points = index_points(points, group_idx)
                # concat these points with the normalized point locations along channels (C)
                # This is why we added +3 channels in the init part of convolutional layers
                grouped_points = torch.cat(
                    [grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            # [B, D, K, S] (batch, features, sampled points under centroids, centroids)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            # Now apply the MLP blocks (since we have 4 dimensions in tensor, we use conv2d blocks instead of conv1d unlike pointnet)
            # We use [i] because this is the ith hierarchical layer in multi scale sampling
            # Remember that this is how our input for mlp_list/conv_blocks looks [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
            # This is the POINT NET PART! (but where the heck is T-Net? idk)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            # Max pool on this layer to get the most prominent features along K dimension and drop dimension
            # [B, D', S] (batch, features, centroids)
            new_points = torch.max(grouped_points, 2)[0]
            # Torch max is done here to get the nearest PROMIENT neighbor for each centroid in new_xyz. (downsampling)
            # append the nearest Prominent neighbors to centroids and its features to our new downsampled set of points for current scale
            new_points_list.append(new_points)
        # Continue loop for all the scales (radius)
        new_xyz = new_xyz.permute(0, 2, 1)
        # Permute centroid tensor back to [B, C, S] format i.e. batch x channels/features x centroids
        # Concatenate all the scales of downsampled points to
        # [M, B, D', S] (where M is number of scales) -> [M*B, D', S] (I think)
        new_points_concat = torch.cat(new_points_list, dim=1)
        # Return centroid xyz locations and the downsampled points feature vector (contains a set of global features which summarize the local features for each centroid)
        # we now have multiscale features for out point cloud at this pointsetabstraction layer (local features)
        return new_xyz, new_points_concat


# PointNet++ encoder which extracts multiscale features from input point cloud.
class PointNetPPEncoder(nn.Module):
    def __init__(self, is_color=True, is_normal=False):
        super(PointNetPPEncoder, self).__init__()
        # Channels accounting for color and or normals for each point
        additional_channels = 0
        if is_color:
            additional_channels += 3
        if is_normal:
            additional_channels += 3
        self.is_color = is_color
        self.is_normal = is_normal
        # Feature Extraction at Multiple Scales inside each set abstraction layer to get multi scale features
        self.sa1 = PointNetSetAbstractionMsg(
            1024, [0.05, 0.1], [16, 32], 3+additional_channels, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(
            256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(
            64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(
            16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])

    def forward(self, xyz):
        B, C, N = xyz.size()
        # split the xyz coords from the color and normals
        if self.is_normal or self.is_color:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]

        # Now send the xyz and points through set abstraction layers
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Now we have multi scale features at every layer of point cloud abstraction
        # We can basically do anything with these learned features
        return l1_points, l2_points, l3_points, l4_points


if __name__ == '__main__':
    import torch
    model = PointNetPPEncoder(True, True)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))
    print(model)
