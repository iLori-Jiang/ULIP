import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import argparse
from tqdm import tqdm

from ULIP_ShapeNet_Dataset.ULIP_ShapeNet import ULIP_ShapeNet
from ULIP_ShapeNet_Dataset.util import encode_text
import models.ULIP_models as models
from main import get_args_parser


def create_fake_args():
    parser = argparse.ArgumentParser('ULIP training and evaluation', parents=[get_args_parser()])
    fake_args = parser.parse_args([])  # 使用空列表初始化命名空间
    # 手动设置每个参数的值
    fake_args.output_dir = './outputs'
    fake_args.npoints = 10000
    fake_args.model = 'ULIP2_PointBERT_Colored'
    fake_args.gpu = 1
    fake_args.test_ckpt_addr = './pretrained_models/ULIP-2-PointBERT-10k-colored-pc-pretrained.pt'
    fake_args.evaluate_3d_ulip2 = True
    fake_args.if_need_clip = True
    fake_args.batch_size = 128

    return fake_args


# 继承现有数据集类
class ULIP_ShapeNet_PC_Embed(ULIP_ShapeNet):
    def __init__(self, keyword):
        super().__init__(keyword=keyword)

    def __getitem__(self, idx):
        data = super().process_index(idx)
        
        pc_embedding = data['pointcloud_embedding_tensor']

        return pc_embedding


def main(keyword):

    args = create_fake_args()

    # ----------------Define class names and descriptions

    shapenet_classes = ['table', 'car', 'bottle', 
                        'chair', 'airplane', 'laptop', 'knife', 'train', 'lamp', 
                        'bin', 'watercraft', 
                        'rocket', 'bag', 
                        'bed', 'display', 'piano', 'telephone', 
                        'bus', 'bowl', 
                        'keyboard', 'guitar', 'bicycle', 'printer', 'cap']
    
    if keyword == 'plane':
        GT_class_index = shapenet_classes.index('airplane')
    else:
        GT_class_index = shapenet_classes.index(keyword)

    pc_descriptions = []

    for i, sn_class in enumerate(shapenet_classes):
        pc_descriptions.append(f'This is a 3D point cloud of a {sn_class}')

    print(pc_descriptions)

    # ---------------- Loading dataset

    # dataset is of point cloud, rendered images, captions of each image, triplet
    dataset = ULIP_ShapeNet_PC_Embed(keyword=keyword)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("")
    print("len(dataset)")
    print(len(dataset))


    # ----------------Model

    ckpt = torch.load(args.test_ckpt_addr, map_location='cpu')

    # 遍历状态字典并移除 'module.' 前缀
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    print("")
    print("=> creating model: {}".format(args.model))

    # create the model
    model = getattr(models, args.model)(args=args)
    model.cuda(args.gpu)

    print("")

    # load the parameters
    model.load_state_dict(state_dict, strict=False)

    print("=> loaded pretrained checkpoint '{}'".format(args.test_ckpt_addr))

    model.eval()

    print("")

    # ---------------- Inference

    pc_class_embeddings = encode_text(pc_descriptions, model, if_normalize=True, gpu=args.gpu, verbose=True)

    total_correct = 0
    total_samples = 0
    logit_diff_stats = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            # to GPU and convert to float
            pointclouds = batch.cuda(args.gpu).float()
            # normalization
            pointclouds = pointclouds / pointclouds.norm(dim=-1, keepdim=True)

            labels = [GT_class_index] * len(batch)
            labels = torch.tensor(labels).cuda(args.gpu)

            logits = pointclouds @ pc_class_embeddings.t()
            predicted_classes = torch.argmax(logits, dim=-1)

            # 计算正确类的logit与其他类logit的差异
            correct_class_logits = logits[torch.arange(len(batch)), labels]
            logit_diffs = correct_class_logits.unsqueeze(1) - logits
            mean_logit_diff = logit_diffs.mean(dim=1)
            logit_diff_stats.append(mean_logit_diff)

            correct = (predicted_classes == labels).sum().item()
            total_correct += correct
            total_samples += len(batch)


    print(f"Current class: {keyword}")

    accuracy = total_correct / total_samples
    print(f"Zero-shot classification accuracy: {accuracy * 100:.2f}%")

    logit_diff_stats = torch.cat(logit_diff_stats)
    print(f"Logit difference statistics: Mean={logit_diff_stats.mean().item()}, Std={logit_diff_stats.std().item()}")


if __name__ == "__main__":
    # keyword = 'plane'
    # keyword = 'chair'
    keyword = 'table'

    main(keyword)


'''
Current class: plane [2860]
Zero-shot classification accuracy: 99.97%
Logit difference statistics: Mean=0.13207530975341797, Std=0.012465103529393673

Current class: chair [4504]
Zero-shot classification accuracy: 99.76%
Logit difference statistics: Mean=0.1647367924451828, Std=0.017219727858901024

Current class: table [4677]
Zero-shot classification accuracy: 99.53%
Logit difference statistics: Mean=0.15189988911151886, Std=0.020195726305246353
'''