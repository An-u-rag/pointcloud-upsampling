import argparse
import os
from data_utils.RandomDataLoader import RandomDataset
from data_utils.S3DISDataLoader import S3DISDataset, S3DISDatasetObjectTest
from models.pointnetpp import PointNetPPEncoder
from models.pointnet import PointNetEncoder
from models.punet import PUnet, UpsampleLoss
import torch
import torch.utils.data as Data
import torch.nn as nn
import datetime
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
VISUAL_DIR = os.path.join("out", "test")
counter = "_objects"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--checkpoint', type=str, default="checkpoints\punet\instant",
                        help='checkpoint dir [default: punet]')
    parser.add_argument('--model', type=str, default='punet',
                        help='model name [default: punet]')
    parser.add_argument('--epochs', type=str, default="[0, 65, 95]",
                        help='epochs to test [default: [0, 65, 95]]')
    parser.add_argument('--batchsize', default=1, type=int,
                        help='Batch size for num pointclouds processed per batch')
    parser.add_argument('--npoint', type=int, default=1024,
                        help='Number of points in Input Point Cloud [default: 1024]')
    parser.add_argument('--upsample_rate', type=int, default=4,
                        help='Rate to upsample at [default: 4]')
    parser.add_argument('--is_color', type=bool, default=False,
                        help='If point in point cloud has r,g,b values [default: False]')
    parser.add_argument('--is_normal', type=bool, default=False,
                        help='If point in point cloud has normal values [default: False]')

    return parser.parse_args()


def visualize(input_data, pred_data, ground_truth_data, model="punet", epoch=0):  # step x batch x n x 3
    input_data = input_data.reshape(-1,
                                    input_data.shape[-2], input_data.shape[-1])
    pred_data = pred_data.reshape(-1, pred_data.shape[-2], pred_data.shape[-1])
    ground_truth_data = ground_truth_data.reshape(
        -1, ground_truth_data.shape[-2], ground_truth_data.shape[-1])  # (step * batch) x n x 3

    # Pick first 5 point clouds to show
    # Iterate through point clouds and display/save as image
    fig, axs = plt.subplots(3, 5, figsize=(
        80, 80), dpi=100, layout="tight", subplot_kw=dict(projection='3d', xticks=[], yticks=[]))
    fig.suptitle(
        'Upsampled Point Clouds (row 1 -> Input, row 2 -> Predictions, row 3 -> Ground Truth)', fontsize=100)

    indices = [0, pred_data.shape[0]//6, pred_data.shape[0] //
               4, pred_data.shape[0]//2, pred_data.shape[0]-1]
    c = 0
    for i in indices:
        x_i = input_data[i, :, 0]
        y_i = input_data[i, :, 1]
        z_i = input_data[i, :, 2]
        x_p = pred_data[i, :, 0]
        y_p = pred_data[i, :, 1]
        z_p = pred_data[i, :, 2]
        x_g = ground_truth_data[i, :, 0]
        y_g = ground_truth_data[i, :, 1]
        z_g = ground_truth_data[i, :, 2]

        sc0 = axs[0, c].scatter(x_i, y_i, z_i)
        axs[0, c].set_title(f"InputPointCloud-{i+1}", fontsize=80)
        sc1 = axs[1, c].scatter(x_p, y_p, z_p)
        axs[1, c].set_title(f"GenPointCloud-{i+1}", fontsize=80)
        sc2 = axs[2, c].scatter(x_g, y_g, z_g)
        axs[2, c].set_title(f"GTPointCloud-{i+1}", fontsize=80)
        c += 1

    savepath = os.path.join(VISUAL_DIR, model + counter, "visuals")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plt.savefig(os.path.join(
        savepath, f"visual_inference_{epoch}"))


def main(args):
    print(device)
    BATCHSIZE = args.batchsize
    CHANNELS = 3
    READ_DIR = args.checkpoint

    if args.is_color:
        CHANNELS += 3
    if args.is_normal:
        CHANNELS += 3

    TEST_DATASET = S3DISDatasetObjectTest(
        num_point=args.npoint, upsample_factor=args.upsample_rate, is_color=False)
    print("Test Dataset Loaded")

    # Feed datasets to dataloader
    test_loader = Data.DataLoader(
        dataset=TEST_DATASET, batch_size=BATCHSIZE, num_workers=4, shuffle=False, drop_last=True)

    epochs = ast.literal_eval(args.epochs)
    for epoch in epochs:
        # Load Model
        if args.model == "pointnetpp" or args.model == "pointnet++":
            # Load PointNet++ Encoder Model for feature extraction
            model = PointNetPPEncoder(
                is_color=False, is_normal=False).to(device)
        elif args.model == "punet":
            # Load PU-Net for point upsampling
            model = PUnet(is_color=args.is_color,
                          is_normal=args.is_normal).to(device)
        else:
            # Load PointNet Encoder Model for feature extraction
            model = PointNetEncoder(in_channels=6).to(device)

        model.load_state_dict(torch.load(os.path.join(
            READ_DIR, f"{args.model}_epoch_{epoch}.pth")))

        # Loss function
        criterion = UpsampleLoss()

        print(f"Starting Inference using epoch {epoch} checkpoint")
        start = time.time()

        # Inference
        with torch.no_grad():
            model.eval()
            gp_list = []
            labels_list = []
            eval_loss_list = []
            input_points = []
            for step, (points, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = points.float().to(device)
                labels = labels.float().to(device)
                points = points.transpose(2, 1)  # B x C x N

                gp = model(points)
                emd_loss, rep_loss = criterion(gp, labels, torch.Tensor(1))
                loss = emd_loss + rep_loss

                eval_loss_list.append(loss.item())
                input_points.append(
                    (points.transpose(2, 1)).cpu().detach().numpy())
                gp_list.append(gp.cpu().detach().numpy())  # steps x B x rN x C
                labels_list.append(labels.cpu().detach().numpy())

            # Call Visualizer function to store output plots of pointclouds for this epoch
            visualize(np.array(input_points), np.array(gp_list), np.array(
                labels_list), model=args.model, epoch=epoch)

        print(
            f"Inference loss: {np.mean(eval_loss_list)}")

        print(
            f"--- Epoch Done in {time.time() - start}. Proceeding to next epoch ---")

        print("-------------Done-------------")


if __name__ == '__main__':
    args = parse_args()
    main(args)
