import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset, S3DISDatasetLarge, S3DISObjectDataset
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
WRITE_DIR = "checkpoints"
VISUAL_DIR = os.path.join("out", "train")
counter = "_object_concatres_higher_repulsion"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='punet',
                        help='model name [default: punet]')
    parser.add_argument('--epochs', default=32, type=int,
                        help='Epochs to run [default: 32]')
    parser.add_argument('--batchsize', default=8, type=int,
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


def visualize(pred_data, ground_truth_data, model="punet", epoch=0):  # step x batch x n x 3
    pred_data = pred_data.reshape(-1, pred_data.shape[-2], pred_data.shape[-1])
    ground_truth_data = ground_truth_data.reshape(
        -1, ground_truth_data.shape[-2], ground_truth_data.shape[-1])  # (step * batch) x n x 3

    # Pick first 5 point clouds to show
    # Iterate through point clouds and display/save as image
    fig, axs = plt.subplots(2, 5, figsize=(
        80, 80), dpi=100, layout="tight", subplot_kw=dict(projection='3d', xticks=[], yticks=[]))
    fig.suptitle(
        'Upsampled Point Clouds (row 1 -> Predictions, row 2 -> Ground Truth)', fontsize=120)
    for i in range(5):
        x_p = pred_data[i, :, 0]
        y_p = pred_data[i, :, 1]
        z_p = pred_data[i, :, 2]
        x_g = ground_truth_data[i, :, 0]
        y_g = ground_truth_data[i, :, 1]
        z_g = ground_truth_data[i, :, 2]

        sc1 = axs[0, i].scatter(x_p, y_p, z_p)
        axs[0, i].set_title(f"PointCloud-{i+1}", fontsize=100)
        sc2 = axs[1, i].scatter(x_g, y_g, z_g)
        axs[1, i].set_title(f"PointCloud-{i+1}", fontsize=100)

    savepath = os.path.join(VISUAL_DIR, model + counter, "visuals")

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    plt.savefig(os.path.join(savepath, f"visual_epoch{epoch}"))


def main(args):
    print(device)
    # train dataset and train loader
    # _Random for now_
    # RandomDataset splits train and test % as 80 and 20 respectively.
    # So the num_pointclouds should be same for both datasets
    BATCHSIZE = args.batchsize
    CHANNELS = 3
    if args.is_color:
        CHANNELS += 3
    if args.is_normal:
        CHANNELS += 3
    TRAIN_DATASET = S3DISObjectDataset(
        train=True, num_point=args.npoint, upsample_factor=args.upsample_rate, test_area=5, is_color=False)
    print("Train Dataset loaded")
    TEST_DATASET = S3DISObjectDataset(
        train=False, num_point=args.npoint, upsample_factor=args.upsample_rate, test_area=5, is_color=False)
    print("Test Dataset Loaded")

    # Feed datasets to dataloader
    train_loader = Data.DataLoader(
        dataset=TRAIN_DATASET, batch_size=BATCHSIZE, num_workers=4, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(
        dataset=TEST_DATASET, batch_size=BATCHSIZE, num_workers=4, shuffle=False, drop_last=True)

    # Load Model
    if args.model == "pointnetpp" or args.model == "pointnet++":
        # Load PointNet++ Encoder Model for feature extraction
        model = PointNetPPEncoder(is_color=False, is_normal=False).to(device)
    elif args.model == "punet":
        # Load PU-Net for point upsampling
        model = PUnet(npoint=args.npoint - (args.npoint//args.upsample_rate), is_color=args.is_color,
                      is_normal=args.is_normal).to(device)
    else:
        # Load PointNet Encoder Model for feature extraction
        model = PointNetEncoder(in_channels=6).to(device)

    # Loss function
    criterion = UpsampleLoss()
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999))

    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch}:")
        start = time.time()
        loss_list = []  # Track of losses of epochs
        gp_list = []  # Track of all the predicted pointclouds
        labels_list = []  # Track of ground truth point clouds
        model.train()
        for step, (points, downsampled_points, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()

            points = points.type(torch.float32).to(
                device)  # B x N' x C (eg: N = 1024)
            downsampled_points = downsampled_points.type(
                torch.float32).to(device)  # B x M x C (eg: M = 768)
            labels = labels.type(torch.float32).to(
                device)  # B x N x C (eg: N' = 4096)

            downsampled_points = downsampled_points.transpose(
                2, 1)  # B x C x N

            # B x rM x C (eg: rM = 3072)
            gp = model(downsampled_points)

            # Concat the predicted point cloud with initial input point cloud in points dimension for geometric struture preservation
            # B x N x C (eg: N = rM + N' = 4096)
            gp = torch.cat((points, gp), dim=1)

            emd_loss, rep_loss = criterion(gp, labels, torch.Tensor(1))

            loss = emd_loss + rep_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            gp_list.append(gp.cpu().detach().numpy())  # steps x B x rN x C
            labels_list.append(labels.cpu().detach().numpy()
                               )  # steps x B x rN x C

        print(f"epoch: {epoch}, loss: {np.mean(loss_list)}")

        if epoch % 1 == 0:
            print("Saving Model....")
            savepath = os.path.join(
                WRITE_DIR, args.model + counter, "instant", f"{args.model}_epoch_{epoch}.pth")
            savepath_for_train = os.path.join(
                WRITE_DIR, args.model + counter, f"{args.model}_epoch_{epoch}.pth")
            if not os.path.exists(os.path.join(WRITE_DIR, args.model + counter, "instant")):
                os.makedirs(os.path.join(
                    WRITE_DIR, args.model + counter, "instant"))
            torch.save(model.state_dict(), savepath)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(loss_list),
            }, savepath_for_train)

            print("Saving predicted point clouds and ground truth point clouds")
            gp_savepath = os.path.join(
                VISUAL_DIR, args.model + counter, "preds")
            gt_savepath = os.path.join(
                VISUAL_DIR, args.model + counter, "ground_truths")
            if not os.path.exists(gp_savepath):
                os.makedirs(gp_savepath)
            if not os.path.exists(gt_savepath):
                os.makedirs(gt_savepath)
            np.save(os.path.join(gp_savepath,
                    f"epoch{epoch}"), np.array(gp_list))
            np.save(os.path.join(gt_savepath,
                    f"epoch{epoch}"), np.array(labels_list))

            # Call Visualizer function to store output plots of pointclouds for this epoch
            visualize(np.array(gp_list), np.array(
                labels_list), model=args.model, epoch=epoch)

        print(f"Staring Evaluation with weights from this epoch")
        # Evaluation at same epoch
        with torch.no_grad():
            model.eval()
            eval_loss_list = []
            for step, (points, downsampled_points, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
                points = points.type(torch.float32).to(
                    device)  # B x N' x C (eg: N = 1024)
                downsampled_points = downsampled_points.type(
                    torch.float32).to(device)  # B x M x C (eg: M = 768)
                labels = labels.type(torch.float32).to(
                    device)  # B x N x C (eg: N' = 4096)

                downsampled_points = downsampled_points.transpose(
                    2, 1)  # B x C x N

                # B x rM x C (eg: rM = 3072)
                gp = model(downsampled_points)

                gp = torch.cat((points, gp), dim=1)
                emd_loss, rep_loss = criterion(gp, labels, torch.Tensor(1))
                loss = emd_loss + rep_loss

                eval_loss_list.append(loss.item())

        print(
            f"Eval done at epoch: {epoch}, eval loss: {np.mean(eval_loss_list)}")

        print(
            f"--- Epoch Done in {time.time() - start}. Proceeding to next epoch ---")

    print("-------------Done-------------")


if __name__ == '__main__':
    args = parse_args()
    main(args)
