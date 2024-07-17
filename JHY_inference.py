from collections import OrderedDict
import models.ULIP_models as models
from main import get_args_parser
import argparse
from tqdm import tqdm

from torch.utils.data import DataLoader
import os

from ULIP_ShapeNet_Dataset.ShapeNet_pointcloud_dataset import PointCloudDataset

import torch


def create_fake_args():
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    fake_args = parser.parse_args([])  # 使用空列表初始化命名空间
    # 手动设置每个参数的值
    fake_args.output_dir = './outputs'
    # fake_args.npoints = 8192
    fake_args.npoints = 10000
    fake_args.model = 'ULIP2_PointBERT_Colored'
    fake_args.gpu = 1
    # fake_args.test_ckpt_addr = './pretrained_models/pointbert_ULIP-2.pt'
    fake_args.test_ckpt_addr = './pretrained_models/ULIP-2-PointBERT-10k-colored-pc-pretrained.pt'
    fake_args.evaluate_3d_ulip2 = True
    fake_args.if_need_clip = False
    fake_args.batch_size = 128

    return fake_args
    

def encode_pointcloud(pc_tensors, model, if_normalize=False, gpu=0, verbose=False):
    '''
    Input:
    pc_tensors: torch tensor of pointcloud [B, N, 3]
    
    Output:
    pc_embeddings: torch tensor encoded [B, encoder_dim]
    '''

    assert len(pc_tensors.shape) == 3

    with torch.no_grad():

        if verbose:
            print("")

        pc_embeddings = model.encode_pc(
                                        pc_tensors.float().cuda(gpu, non_blocking=True)
                                        ).float()
        
        if if_normalize:
            pc_embeddings = pc_embeddings / pc_embeddings.norm(dim=-1, keepdim=True)

        if verbose:
            print("pc_embeddings.shape")
            print(pc_embeddings.shape)
            print("pc_embeddings.norm(dim=-1)")
            print(pc_embeddings.norm(dim=-1))

            print("")
        
        return pc_embeddings
    

def main(path_data_pc, path_data_pc_embeddings, if_test=True):

    # ----------------------- model part

    args = create_fake_args()

    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')

    # 遍历状态字典并移除 'module.' 前缀
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("=> creating model: {}".format(args.model))

    # create the model
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    # load the parameters
    model.load_state_dict(state_dict, strict=False)
    print("=> loaded pretrained checkpoint '{}'".format(args.test_ckpt_addr))

    model.eval()

    # ----------------------- data part

    dataset = PointCloudDataset(path_data_pc, if_need_rgb=True, if_xyz_rgb=True, verbose=if_test)

    if if_test:
        pc_tensor_0, _ = dataset[0]
        pc_tensor_1, _ = dataset[1]
        pc_tensor_2, _ = dataset[2]
        pc_tensor_3, _ = dataset[3]

        batch_tensor = torch.cat((pc_tensor_0, pc_tensor_1, pc_tensor_2, pc_tensor_3), dim=0)

        embeddings = encode_pointcloud(batch_tensor, model, if_normalize=True, gpu=args.gpu, verbose=if_test)

        print("")
        print("Process finish")
        print("embeddings.shape")
        print(embeddings.shape)

    else:

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        if not os.path.exists(path_data_pc_embeddings):
            os.makedirs(path_data_pc_embeddings)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                point_clouds, file_names = batch
                if len(point_clouds.shape) == 4 and point_clouds.shape[1] == 1:
                    point_clouds = point_clouds.squeeze(1)  # 从 [batch_size, 1, N, 3] 变为 [batch_size, N, 3]
                    
                embeddings = encode_pointcloud(point_clouds, model, if_normalize=False, gpu=args.gpu, verbose=if_test)
                
                # 逐个保存嵌入向量到单独的文件
                for embedding, file_name in zip(embeddings, file_names):
                    save_path = os.path.join(path_data_pc_embeddings, f"{file_name}.pt")
                    torch.save(embedding.cpu(), save_path)
                    
        print("")
        print("All the files are encoded and saved successfully")
        
        output_files_len = len(os.listdir(path_data_pc_embeddings))
        print(f"Totally {output_files_len} files are saved")

        print(f"Compared to the number of dataset: {len(dataset)}")



if __name__ == "__main__":

    path_data = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets"
    path_data_pc = os.path.join(path_data, "shapenet_pc")
    path_data_pc_embeddings = os.path.join(path_data, "ulip_pc_embeddings")
    if_test = False

    main(path_data_pc, path_data_pc_embeddings, if_test=if_test)

    