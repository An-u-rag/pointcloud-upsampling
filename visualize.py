import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm

READ_DIR = "out"
WRITE_DIR = os.path.join("out", "visuals")


def main(model="punet", epoch=0):
    pred_path = os.path.join(READ_DIR, "preds", model, f"epoch{epoch}.npy")
    ground_truth_path = os.path.join(
        READ_DIR, "ground_truths", model, f"epoch{epoch}.npy")

    pred_data = np.load(pred_path)  # step x batch x n x 3
    pred_data = pred_data.reshape(-1, pred_data.shape[-2], pred_data.shape[-1])
    ground_truth_data = np.load(ground_truth_path)
    ground_truth_data = ground_truth_data.reshape(
        -1, ground_truth_data.shape[-2], ground_truth_data.shape[-1])

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

        # fig1 = plt.figure(figsize=(8, 8))
        # ax = fig1.add_subplot(111, projection='3d')

        # ax.scatter(x_p, y_p, z_p)

        # fig2 = plt.figure(figsize=(8, 8))
        # ax = fig2.add_subplot(111, projection='3d')

        # ax.scatter(x_g, y_g, z_g)

        sc1 = axs[0, i].scatter(x_p, y_p, z_p, cmap=cm.jet)
        axs[0, i].set_title(f"PointCloud-{i+1}", fontsize=100)
        sc2 = axs[1, i].scatter(x_g, y_g, z_g, cmap=cm.jet)
        axs[1, i].set_title(f"PointCloud-{i+1}", fontsize=100)

    if not os.path.exists(WRITE_DIR):
        os.makedirs(WRITE_DIR)

    savepath = os.path.join(WRITE_DIR, model)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"visual_epoch{epoch}"))


if __name__ == '__main__':
    epoch_range = [0, 100]
    model = "punet"
    for e in range(epoch_range[0], epoch_range[1], 5):
        main(model, e)
