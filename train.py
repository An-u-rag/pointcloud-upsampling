import argparse
import os
from data_utils.RandomDataLoader import RandomDataset
from models.pointnetpp import PointNetPPEncoder
from models.pointnet import PointNetEncoder
import torch
import torch.utils.data as Data
import torch.nn as nn
import datetime
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnetpp',
                        help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--epochs', default=32, type=int,
                        help='Epochs to run [default: 32]')
    parser.add_argument('--batchsize', default=8, type=int,
                        help='Batch size for num pointclouds processed per batch')
    parser.add_argument('--npoint', type=int, default=4096,
                        help='Point Number [default: 4096]')

    return parser.parse_args()


def main(args):
    # train dataset and train loader
    # _Random for now_
    # RandomDataset splits train and test % as 80 and 20 respectively.
    # So the num_pointclouds should be same for both datasets
    BATCHSIZE = args.batchsize
    TRAIN_DATASET = RandomDataset(
        train=True, num_pointclouds=30, num_point=4092, channels=6)
    print("Train Dataset loaded")
    TEST_DATASET = RandomDataset(
        train=False, num_pointclouds=30, num_point=4092, channels=6)
    print("Test Dataset Loaded")
    # Feed datasets to dataloader
    train_loader = Data.DataLoader(
        dataset=TRAIN_DATASET, batch_size=BATCHSIZE, num_workers=4, shuffle=True)
    test_loader = Data.DataLoader(
        dataset=TEST_DATASET, batch_size=BATCHSIZE, num_workers=4, shuffle=False)

    # Load Model
    if args.model == "pointnetpp" or args.model == "pointnet++":
        # Load PointNet++ Encoder Model for feature extraction
        model = PointNetPPEncoder(is_color=True, is_normal=False).to(device)
    else:
        # Load PointNet Encoder Model for feature extraction
        model = PointNetEncoder(in_channels=6).to(device)

    # Loss function - None as of now but once we add a downstream model, this will be populated

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch}:")
        start = time.time()
        model.train()
        for step, points in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            points = points.data.numpy()  # B x N x C
            points = torch.Tensor(points)
            points = points.float().to(device)
            points = points.transpose(2, 1)  # B x C x N

            # Get features for each point cloud at each set abstraction layer
            pc_multiscale_features = model(points)

            optimizer.step()

        print(f"Extracted features at epoch {epoch}")
        print(f"Staring Evaluation with weights from this epoch")
        # Evaluation at same epoch
        with torch.no_grad():
            model.eval()

            for step, points in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = points.data.numpy()  # B x N x C
                points = torch.Tensor(points)
                points = points.float().to(device)
                points = points.transpose(2, 1)  # B x C x N

                pc_multiscale_features = model(points)

        print("Eval done ---")

        print(
            f"Epoch Done in {time.time() - start}. Proceeding to next epoch ---")

    print("-------------Done-------------")


if __name__ == '__main__':
    args = parse_args()
    main(args)
