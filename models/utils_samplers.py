import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C] (eg: [B, 1024, 3])
        dst: target points, [B, M, C] (eg: [B, 512, 3])
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, C = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # diff = src.unsqueeze(2).expand(B, N, M, C) - \
    #     dst.unsqueeze(1).expand(B, N, M, C)
    # dist = torch.sum(diff ** 2, -1)
    dist = torch.cdist(src, dst, p=2.0)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C] (eg: [B, 1024, 3])
        idx: sample index data, [B, S] (eg: B, 1024)
    Return:
        new_points:, indexed points data, [B, S, C] 
    """
    S = idx.size(1)
    if idx.device != "cpu":
        # print("Entered1: ", idx[0, 0])
        idx = torch.where(idx > S-1, S-1, idx).to(idx.device)
        # print("Entered2: ", idx[0, 0])
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].type(distance.dtype)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(
    #     device).view(1, 1, N).repeat([B, S, 1])  # [B, 1024, 1024] (contains indices up to N=1024)
    sqrdists = square_distance(new_xyz, xyz)  # [B, 1024, 1024]
    # group_idx[sqrdists > radius ** 2] = N # If radius is too small, then all the indices are set to 1024 ###

    # Get the sorted distances for each point in 1024
    sort_dis, group_idx = sqrdists.sort(dim=-1)
    # Check if distance is greater than radius, if so, drop it to max index
    group_idx[sort_dis > radius**2] = N
    # Get first 32 indices from sorted distance array
    group_idx = group_idx[:, :, :nsample]
    group_first = group_idx[:, :, 0].view(
        B, S, 1).repeat([1, 1, nsample])  # reshape
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # mask = group_idx == N
    # group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: (eg: 1024)
        radius:
        nsample: (eg: 32)
        xyz: input points position data, [B, N, 3] (eg: [B, 1024, 3])
        points: input points data, [B, N, D] (eg: [B, 1024, 9])
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3] (eg: [B, 1024, 32, 3])
        new_points: sampled points data, [B, npoint, nsample, 3+D] (eg: [B, 1024, 32, 3+9])
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, 1024, 3]
    # print("fps_idx: ", fps_idx.size())  # [B, 1024]
    new_xyz = index_points(xyz, fps_idx)  # [B, 1024, 3], [B, 1024]
    # print("new_xyz: ", new_xyz.size())  # [B, 1024, 3]
    # [B, 1024, 3], [B, 1024, 3]
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # print("idx: ", idx.size())  # [B, 1024, 32]
    # [B, npoint, nsample, C] = [B, 1024, 3], [B, 1024, 32]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        # [B, npoint, nsample, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
