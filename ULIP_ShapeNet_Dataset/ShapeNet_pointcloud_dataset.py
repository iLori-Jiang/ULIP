import os
import torch
from torch.utils.data import Dataset
import numpy as np

from .util import pc_numpy2tensor, process_pointcloud


class PointCloudDataset(Dataset):
    def __init__(self, path_data_pc, if_need_rgb=False, if_xyz_rgb=True, verbose=False):
        self.path_data_pc = path_data_pc
        self.all_files = sorted(os.listdir(path_data_pc))

        self.if_need_rgb = if_need_rgb
        self.if_xyz_rgb = if_xyz_rgb
        self.verbose = verbose

    
    def __len__(self):
        return len(self.all_files)
    

    def __getitem__(self, idx):
        # 读取点云文件
        file_name = self.all_files[idx]
        pc_np = np.load(os.path.join(self.path_data_pc, file_name))
        
        # 将 numpy 数组转换为 tensor 并添加批次维度
        pc_tensor = pc_numpy2tensor(pc_np)

        pc_tensor = process_pointcloud(pc_tensor, if_need_rgb=self.if_need_rgb, if_xyz_rgb=self.if_xyz_rgb, verbose=self.verbose)
        
        return pc_tensor, file_name[:-4]
    

if __name__ == "__main__":
    path_data = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets"
    path_data_pc = os.path.join(path_data, "shapenet_pc")

    dataset = PointCloudDataset(path_data_pc, if_need_rgb=True, verbose=True)

    print("len(dataset)")
    print(len(dataset))

    index = 5
    print("index")
    print(index)

    pc_tensor, file_name = dataset[index]

    print(pc_tensor.shape)
    print(file_name)
