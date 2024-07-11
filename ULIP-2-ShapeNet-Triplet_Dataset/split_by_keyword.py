import json
import os
from tqdm import tqdm
from collections import defaultdict
import logging

'''
OUTPUT

file name: plane.json   [keyword]

file:
[
    "02691156-10155655850468db78d106ce0a280f87",    [point cloud name]
    "02691156-1021a0914a7207aff927ed529ad90a11",
    "02691156-1026dd1b26120799107f68a9cb8e3c",
    "02691156-103c9e43cdf6501c62b600da24e0965",
    .
    .
    .
]
'''


def load_captions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def filter_by_keyword(captions_data, keywords, single_caption_threshold=5):
    filtered_data = {keyword: [] for keyword in keywords}
    
    for img_path, captions in tqdm(captions_data.items(), desc="Filtering captions"):
        if img_path.endswith("_depth0001.png"):
            continue  # 跳过深度图像文件

        # 提取点云文件名
        pointcloud_name = img_path.split('/')[-2]
        keyword_count = defaultdict(int)
        
        for caption in captions:
            for keyword in keywords:
                if keyword in caption.lower():
                    keyword_count[keyword] += 1
        
        for keyword in keywords:
            if keyword_count[keyword] >= single_caption_threshold:
                filtered_data[keyword].append(pointcloud_name)
    
    return filtered_data

def refine_filtered_data(filtered_data, image_threshold):
    refined_data = {keyword: [] for keyword in filtered_data.keys()}
    
    for keyword, pointclouds in filtered_data.items():
        pointcloud_count = defaultdict(int)
        
        for pointcloud in pointclouds:
            pointcloud_count[pointcloud] += 1
        
        for pointcloud, count in pointcloud_count.items():
            if count >= image_threshold:
                refined_data[keyword].append(pointcloud)
    
    return refined_data

def save_filtered_data(filtered_data, output_dir):
    for keyword, files in filtered_data.items():
        output_path = os.path.join(output_dir, f"{keyword}.json")
        with open(output_path, 'w') as f:
            json.dump(files, f, indent=4, ensure_ascii=False)
        logging.info(f"Saved {len(files)} items for keyword '{keyword}' to {output_path}")


def main(json_path, output_dir, keywords, single_caption_threshold, image_threshold):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志记录到文件
    log_file = os.path.join(output_dir, "log.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    # 记录阈值信息
    logging.info(f"single_caption_threshold = {single_caption_threshold}")
    logging.info(f"image_threshold = {image_threshold}")

    # 载入描述数据
    logging.info(f"Loading captions from {json_path}")
    captions_data = load_captions(json_path)

    # 初次过滤数据
    logging.info("Initial filtering of captions based on keywords")
    filtered_data = filter_by_keyword(captions_data, keywords, single_caption_threshold)

    # 细化过滤结果
    logging.info("Refining filtered data based on image threshold")
    refined_data = refine_filtered_data(filtered_data, image_threshold)

    # 保存最终结果
    logging.info("Saving refined filtered data")
    save_filtered_data(refined_data, output_dir)
    logging.info("Processing completed")

if __name__ == "__main__":
    # 参数设置
    dataset_path = "/mnt/disk2/iLori/ShapeNet-55-ULIP-2-triplets/"
    json_path = os.path.join(dataset_path, "ULIP-shapenet_triplets_captions.json")
    output_dir = os.path.join(dataset_path, "filter_by_keyword")
    
    keywords = ["chair", "plane", "table"]
    single_caption_threshold = 5
    image_threshold = 30
    
    main(json_path, output_dir, keywords, single_caption_threshold, image_threshold)
