import json
import os
from tqdm import tqdm

def load_captions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
    

def split_json_captions(json_path, output_dir):
    captions_data = load_captions(json_path)
    
    organized_data = {}
    
    # 使用tqdm显示进度条
    for img_path, captions in tqdm(captions_data.items(), desc="Organizing captions"):
        pointcloud_name = os.path.basename(img_path).split('_r_')[0]
        if pointcloud_name not in organized_data:
            organized_data[pointcloud_name] = {}
        img_name = os.path.basename(img_path)
        organized_data[pointcloud_name][img_name] = captions
    
    # 保存时也使用tqdm显示进度条
    for pointcloud, captions in tqdm(organized_data.items(), desc="Saving captions"):
        output_path = os.path.join(output_dir, f"{pointcloud}.json")
        with open(output_path, 'w') as f:
            json.dump(captions, f, indent=4)


# 路径定义
path_json = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets/ULIP-shapenet_triplets_captions.json"
output_dir = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets/captions"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 拆分并保存JSON文件
split_json_captions(path_json, output_dir)