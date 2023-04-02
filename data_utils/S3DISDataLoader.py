from torch.utils.data import Dataset
import torch
import os
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

DATA_DIR = "data/pointclouds/s3dis_npy"
OBJECT_TEST_DIR = "../data/pointclouds/s3dis_object_tests"


class S3DISDataset(Dataset):
    def __init__(self, train=True, num_point=1024, upsample_factor=4, test_area=5, is_color=False, patch_radius=0.1, num_patch=5):
        super().__init__()
        self.num_point = num_point
        self.is_color = is_color
        self.channels = 6 if is_color else 3
        self.upsample_factor = upsample_factor
        self.d = patch_radius
        self.M = num_patch
        self.train = train

        # Read from data directory and add point cloud file name to an array
        rooms = sorted(os.listdir(DATA_DIR))
        rooms = [room for room in rooms if 'Area_' in room]

        # Split based on flag
        if self.train:
            rooms_split = [
                room for room in rooms if not f'Area_{test_area}' in room]
        else:
            rooms_split = [
                room for room in rooms if f'Area_{test_area}' in room]

        # Read point cloud data from files
        start = time.time()
        pointclouds = []
        for room in tqdm(rooms_split):
            if room.split('.')[1] == "npy":
                points = np.load(os.path.join(DATA_DIR, room))
                points = points[:, :self.channels]
            else:
                with open(os.path.join(DATA_DIR, room), 'r') as f:
                    lines = f.readlines()
                    points = np.zeros((len(lines), self.channels))
                    for i in range(len(lines)):
                        point = lines[i].strip().split(' ')
                        points[i, :] = [float(point[j])
                                        for j in range(self.channels)]
            pointclouds.append(points)

        print(f"Extracted in {time.time() - start}")

        # Sample patch (num_point * upsample_factor) from each point cloud
        # For now lets just randomly sample using uniform dist. (realistically we have to use furthest point sample or grid sampling)
        # for i, pc in enumerate(pointclouds):
        #     sample_idx = np.random.choice(
        #         len(pc), self.num_point * self.upsample_factor, replace=False)
        #     pointclouds[i] = pc[sample_idx]

        # # Now each point cloud has numpoint * upsample_factor number of points
        # self.pointclouds = np.array(pointclouds)

        # Sample patch (num_point * upsample_factor) from each point cloud
        patches = []
        print(
            f"////////////////////Extracting {self.M} Patches from each Point Cloud////////////////////")
        for i, pc in tqdm(enumerate(pointclouds), total=len(pointclouds)):
            patch_centers = pc[np.random.choice(
                len(pc), self.M, replace=False)]
            for patch_center in patch_centers:
                patch = self.get_patch(pc, patch_center)
                patches.append(patch)

        # Now each patch has patch_point_num number of points
        self.pointclouds = np.array(patches)

        # Should have Rooms x Num_point * upsample_factor * channels
        print(f"Point clouds patch extraction done: {self.pointclouds.shape}")

        # Now Normalize the coordinates with their respective min and max values
        # TO Normalize we need to find max and min coordinates of each point cloud
        # (xi – min(x)) / (max(x) – min(x))

        print("////////////////////Working on Patch Normalization////////////////////")
        for i, pc in tqdm(enumerate(self.pointclouds), total=len(self.pointclouds)):
            self.pointclouds[i, :, 0] = (
                pc[:, 0] - np.amin(pc[:, 0])) / (np.amax(pc[:, 0]) - np.amin(pc[:, 0]))
            self.pointclouds[i, :, 1] = (
                pc[:, 1] - np.amin(pc[:, 1])) / (np.amax(pc[:, 1]) - np.amin(pc[:, 1]))
            self.pointclouds[i, :, 2] = (
                pc[:, 2] - np.amin(pc[:, 2])) / (np.amax(pc[:, 2]) - np.amin(pc[:, 2]))

        # Now that we have normalized x, y, z coordinates, we can store it and call it in getitem

    def get_patch(self, pc, patch_center):  # pc -> N x 3, patch_center -> 1 x 3
        dists = np.linalg.norm(pc-patch_center, axis=1)
        # Check if distance is more than radius of patch then set to max
        dists[dists > self.d] = np.amax(dists)
        patch_indices = np.argsort(dists)  # N indices
        patch = pc[patch_indices]  # N x 3
        # Get first num_point * upsample_factor indices
        patch = patch[:self.num_point * self.upsample_factor]  # N' x 3
        patch = patch - patch_center  # Centroid to 0,0,0
        return patch  # N' x 3

    def __getitem__(self, i):
        # We have generated the upsampled point cloud with 4096 points
        # The upsampling factor is r=4
        # Therefore out input will be downscaled to 4096/4 = 1024 and label with the entire pointcloud

        label = self.pointclouds[i]  # 4096 x C
        input_idx = np.random.choice(
            self.num_point * self.upsample_factor, self.num_point, replace=False)
        input = label[input_idx]
        return input, label

    def __len__(self):
        return len(self.pointclouds)


class S3DISDatasetObjectTest(Dataset):
    def __init__(self, num_point=1024, upsample_factor=4, is_color=False):
        super().__init__()
        self.num_point = num_point
        self.is_color = is_color
        self.channels = 6 if is_color else 3
        self.upsample_factor = upsample_factor

        # Read from data directory and add point cloud file name to an array
        objects = sorted(os.listdir(OBJECT_TEST_DIR))
        objects = [object for object in objects if 'Object_' in object]

        # Read point cloud data from files
        start = time.time()
        pointclouds = []
        for object in tqdm(objects):
            if object.split('.')[1] == "npy":
                points = np.load(os.path.join(OBJECT_TEST_DIR, object))
                points = points[:, :self.channels]
            else:
                with open(os.path.join(OBJECT_TEST_DIR, object), 'r') as f:
                    lines = f.readlines()
                    points = np.zeros(
                        (self.num_point * self.upsample_factor, self.channels))
                    # Downscale to 4096 points
                    indices = np.random.choice(
                        len(lines), self.num_point * self.upsample_factor, replace=False)
                    for i, index in enumerate(indices):
                        point = lines[index].strip().split(' ')
                        points[i, :] = [float(point[j])
                                        for j in range(self.channels)]
                    print(len(points))
            pointclouds.append(points)

        print(f"Extracted in {time.time() - start}")

        self.pointclouds = np.array(pointclouds)

        print("min: ", np.amin(self.pointclouds[0], axis=0))
        print("max: ", np.amax(self.pointclouds[0], axis=0))

        self.visualize(self.pointclouds)

        # Centroid Normalization
        for i in range(len(self.pointclouds)):
            pc = self.pointclouds[i]
            length = pc.shape[0]
            sum_x = np.sum(pc[:, 0])
            sum_y = np.sum(pc[:, 1])
            sum_z = np.sum(pc[:, 2])
            centroid = np.array([sum_x/length, sum_y/length, sum_z/length])
            print("centroid: ", centroid)
            self.pointclouds[i] = pc - centroid

        print("min: ", np.amin(self.pointclouds[0], axis=0))
        print("max: ", np.amax(self.pointclouds[0], axis=0))

        self.visualize(self.pointclouds)

        # Now Normalize the coordinates with their respective min and max values
        # TO Normalize we need to find max and min coordinates of each point cloud
        # (xi – min(x)) / (max(x) – min(x))
        print(
            "////////////////////Working on Object Patch Normalization////////////////////")
        for i, pc in tqdm(enumerate(self.pointclouds), total=len(self.pointclouds)):
            self.pointclouds[i, :, 0] = np.uint8((
                pc[:, 0] - np.amin(pc[:, 0])) / (np.amax(pc[:, 0]) - np.amin(pc[:, 0])) * 255)
            self.pointclouds[i, :, 1] = np.uint8((
                pc[:, 1] - np.amin(pc[:, 1])) / (np.amax(pc[:, 1]) - np.amin(pc[:, 1])) * 255)
            self.pointclouds[i, :, 2] = np.uint8((
                pc[:, 2] - np.amin(pc[:, 2])) / (np.amax(pc[:, 2]) - np.amin(pc[:, 2])) * 255)

        # Now that we have normalized x, y, z coordinates, we can store it and call it in getitem
        print("min: ", np.amin(self.pointclouds[0], axis=0))
        print("max: ", np.amax(self.pointclouds[0], axis=0))

        self.visualize(self.pointclouds)

    def visualize(self, data):  # step x batch x n x 3
        for i in range(len(data)):
            x = data[i, :, 0]
            y = data[i, :, 1]
            z = data[i, :, 2]

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x, y, z)
            plt.show()

    def __getitem__(self, i):
        # We have generated the upsampled point cloud with 4096 points
        # The upsampling factor is r=4
        # Therefore out input will be downscaled to 4096/4 = 1024 and label with the entire pointcloud

        label = self.pointclouds[i]  # 4096 x C
        input_idx = np.random.choice(
            self.num_point * self.upsample_factor, self.num_point, replace=False)
        input = label[input_idx]
        self.visualize(input.reshape(-1, input.shape[0], input.shape[1]))

        label = torch.as_tensor(label, dtype=torch.uint8).cuda()
        input = torch.as_tensor(input, dtype=torch.uint8).cuda()
        print("Get label size: ", label.dtype,
              label.element_size() * label.nelement())
        print("Get input size: ", input.dtype,
              input.element_size() * input.nelement())

        return input, label

    def __len__(self):
        return len(self.pointclouds)


class S3DISDatasetLarge(Dataset):
    def __init__(self, train=True, num_pointclouds=30, num_point=10000, upsample_factor=4, test_area=5, is_color=False, patch_radius=0.1, num_patch=5):
        super().__init__()
        self.num_point = num_point
        self.is_color = is_color
        self.channels = 6 if is_color else 3
        self.upsample_factor = upsample_factor
        self.d = patch_radius
        self.M = num_patch
        self.train = train

        # Read from data directory and add point cloud file name to an array
        rooms = sorted(os.listdir(DATA_DIR))
        rooms = [room for room in rooms if 'Area_' in room]

        # Split based on flag
        if self.train:
            rooms_split = [
                room for room in rooms if not f'Area_{test_area}' in room]
            rooms_split = rooms_split[:num_pointclouds]
        else:
            rooms_split = [
                room for room in rooms if f'Area_{test_area}' in room]
            rooms_split = rooms_split[:num_pointclouds//5]

        # Read point cloud data from files
        start = time.time()
        pointclouds = []
        for room in tqdm(rooms_split):
            if room.split('.')[1] == "npy":
                points = np.load(os.path.join(DATA_DIR, room))
                points = points[:, :self.channels]
            else:
                with open(os.path.join(DATA_DIR, room), 'r') as f:
                    lines = f.readlines()
                    points = np.zeros((len(lines), self.channels))
                    for i in range(len(lines)):
                        point = lines[i].strip().split(' ')
                        points[i, :] = [float(point[j])
                                        for j in range(self.channels)]
            pointclouds.append(points)

        print(f"Extracted in {time.time() - start}")

        # Sample patch (num_point * upsample_factor) from each point cloud
        # For now lets just randomly sample using uniform dist. (realistically we have to use furthest point sample or grid sampling)
        # for i, pc in enumerate(pointclouds):
        #     sample_idx = np.random.choice(
        #         len(pc), self.num_point * self.upsample_factor, replace=False)
        #     pointclouds[i] = pc[sample_idx]

        # # Now each point cloud has numpoint * upsample_factor number of points
        # self.pointclouds = np.array(pointclouds)

        # Sample patch (num_point * upsample_factor) from each point cloud
        patches = []
        print(
            f"////////////////////Extracting {self.M} Patches from each Point Cloud////////////////////")
        for i, pc in tqdm(enumerate(pointclouds), total=len(pointclouds)):
            patch_centers = pc[np.random.choice(
                len(pc), self.M, replace=False)]
            for patch_center in patch_centers:
                patch = self.get_patch(pc, patch_center)
                patches.append(patch)

        # Now each patch has patch_point_num number of points
        self.pointclouds = np.array(patches)

        # Should have Rooms x Num_point * upsample_factor * channels
        print(f"Point clouds patch extraction done: {self.pointclouds.shape}")

        # Now Normalize the coordinates with their respective min and max values
        # TO Normalize we need to find max and min coordinates of each point cloud
        # (xi – min(x)) / (max(x) – min(x))

        print("////////////////////Working on Patch Normalization////////////////////")
        for i, pc in tqdm(enumerate(self.pointclouds), total=len(self.pointclouds)):
            self.pointclouds[i, :, 0] = np.uint8((
                pc[:, 0] - np.amin(pc[:, 0])) / (np.amax(pc[:, 0]) - np.amin(pc[:, 0])) * 255)
            self.pointclouds[i, :, 1] = np.uint8((
                pc[:, 1] - np.amin(pc[:, 1])) / (np.amax(pc[:, 1]) - np.amin(pc[:, 1])) * 255)
            self.pointclouds[i, :, 2] = np.uint8((
                pc[:, 2] - np.amin(pc[:, 2])) / (np.amax(pc[:, 2]) - np.amin(pc[:, 2])) * 255)

        # Now that we have normalized x, y, z coordinates, we can store it and call it in getitem

    def get_patch(self, pc, patch_center):  # pc -> N x 3, patch_center -> 1 x 3
        dists = np.linalg.norm(pc-patch_center, axis=1)
        # Check if distance is more than radius of patch then set to max
        dists[dists > self.d] = np.amax(dists)
        patch_indices = np.argsort(dists)  # N indices
        patch = pc[patch_indices]  # N x 3
        # Get first num_point * upsample_factor indices
        patch = patch[:self.num_point * self.upsample_factor]  # N' x 3
        patch = patch - patch_center  # Centroid to 0,0,0
        return patch  # N' x 3

    def __getitem__(self, i):
        # We have generated the upsampled point cloud with 4096 points
        # The upsampling factor is r=4
        # Therefore out input will be downscaled to 4096/4 = 1024 and label with the entire pointcloud

        label = self.pointclouds[i]  # 4096 x C
        input_idx = np.random.choice(
            self.num_point * self.upsample_factor, self.num_point, replace=False)
        input = label[input_idx]

        label = torch.as_tensor(label, dtype=torch.float32).cuda()
        input = torch.as_tensor(input, dtype=torch.float32).cuda()

        return input, label

    def __len__(self):
        return len(self.pointclouds)


if __name__ == '__main__':
    import torch
    import torch.utils.data as Data

    data = S3DISDatasetObjectTest()

    data.__getitem__(0)
