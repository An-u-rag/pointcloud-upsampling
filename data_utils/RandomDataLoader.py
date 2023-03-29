from torch.utils.data import Dataset
import os
import numpy as np

# generate random string
A, Z = np.array(["A", "Z"]).view("int32")
LEN = 5
ranstr = np.random.randint(
    low=A, high=Z, size=LEN, dtype="int32").view(f"U{LEN}")[0]
WRITE_DIR = f"data/pointclouds/randomgen/data_{ranstr}"


class RandomDataset(Dataset):
    def __init__(self, train=True, num_pointclouds=30, num_point=1024, upsample_factor=4, channels=3):
        super().__init__()
        self.num_pointclouds = num_pointclouds
        self.num_point = num_point
        self.channels = channels
        self.upsample_factor = upsample_factor
        # generate random point clouds with 2048 points each with X,Y,Z,R,G,B
        # Both XYZ and RGB are already normalized since both lie in [0,1) range
        self.train = train
        if self.train:
            self.data = np.random.rand(
                int(self.num_pointclouds * 0.8), self.num_point*self.upsample_factor, self.channels)  # B x N x C
        else:
            self.data = np.random.rand(
                int(self.num_pointclouds * 0.2), self.num_point*self.upsample_factor, self.channels)

        # Write the randomly generated point cloud data to a text file
        # Check if WRITE_DIR exists and make if not
        if not os.path.exists(WRITE_DIR):
            os.makedirs(WRITE_DIR)
        # Write to a file (Huge time sink)

        for i, pc in enumerate(self.data):
            # print(f"Writing to point cloud file : data_{ranstr}_{i}")
            with open(f'{WRITE_DIR}/pc_{i}.txt', 'w') as f:
                for n, p in enumerate(pc):
                    point_line = ''
                    for c in p:
                        point_line += str(c)
                        point_line += " "
                    f.write(f'{n} {point_line}')
                    f.write('\n')

    def __getitem__(self, i):
        # We have generated the upsampled point cloud with 4096 points
        # The upsampling factor is r=4
        # Therefore out input will be downscaled to 4096/4 = 1024 and label with the entire pointcloud

        label = self.data[i]  # 4096 x C
        input_idx = np.random.choice(
            self.num_point * self.upsample_factor, self.num_point, replace=False)
        input = label[input_idx]
        return input, label

    def __len__(self):
        return len(self.data)
