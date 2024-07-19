import torch


def pc_numpy2tensor(pc_np):
    '''
    input numpy array:  [N, 3]
    return tensor:      [1, N, 3]
    '''
    return torch.from_numpy(pc_np).unsqueeze(0)


def check_normalization(pc_tensor):

    '''
    pc_tensor: [1, N, 3]
    '''

    min_vals = torch.min(pc_tensor, dim=1).values
    max_vals = torch.max(pc_tensor, dim=1).values
    mean_vals = torch.mean(pc_tensor, dim=1)

    range_normalized = False
    center_normalized = False

    if torch.all(min_vals >= 0) and torch.all(max_vals <= 1):
        range_normalized = True
        range_info = "The point cloud range is normalized to [0, 1]."
    elif torch.all(min_vals >= -1) and torch.all(max_vals <= 1):
        range_normalized = True
        range_info = "The point cloud range is normalized to [-1, 1]."
    else:
        range_info = "The point cloud range is not normalized."

    if torch.allclose(mean_vals, torch.zeros_like(mean_vals), atol=1e-1):
        center_normalized = True
        center_info = "The point cloud is centered at the origin."
    else:
        center_info = f"The point cloud is centered at {mean_vals.mean(dim=0).tolist()}."

    return range_info, center_info, range_normalized, center_normalized



def normalize_point_cloud(point_cloud, range_normalized, center_normalized, verbose=False):
    """
    归一化点云，将其中心移动到原点，并缩放到单位球体内。

    参数:
    point_cloud (torch.Tensor): 形状为 [1, N, 3] 的点云张量。

    返回:
    torch.Tensor: 归一化后的点云张量。
    """
    # 检查输入张量的形状
    assert point_cloud.shape[0] == 1 and point_cloud.shape[2] == 3, "输入点云的形状应为 [1, N, 3]"

    # 移除批次维度
    point_cloud = point_cloud.squeeze(0)

    if not center_normalized:
        # 计算点云的质心
        centroid = point_cloud.mean(dim=0, keepdim=True)

        # 将点云的质心移动到原点
        point_cloud = point_cloud - centroid

        if verbose:
            print("Normalizing the center...")
            print("Center after normalization:")
            print(point_cloud.mean(dim=0))

    if not range_normalized:
        # 计算点云到原点的最大距离
        max_distance = torch.max(torch.norm(point_cloud, dim=1))

        # 将点云缩放到单位球体内
        point_cloud = point_cloud / max_distance

        if verbose:
            print("Normalizing the range...")
            print("Range after normalization:")
            print(torch.max(torch.norm(point_cloud, dim=1)))

    # 恢复批次维度
    point_cloud = point_cloud.unsqueeze(0)

    return point_cloud


def process_pointcloud(pc_xyz_tensor, if_need_rgb=False, if_xyz_rgb=True, verbose=False):
    '''
    Input: 
    pc_xyz_tensor: torch tensor [1, points, 3]

    Output: 
    pc_tensor: torch tensor [1, points, 3]
    '''

    assert len(pc_xyz_tensor.shape) == 3

    if verbose:
        print("")

        print("pc_xyz_tensor.shape")
        print(pc_xyz_tensor.shape)

    range_info, center_info, range_normalized, center_normalized = check_normalization(pc_xyz_tensor)

    if verbose:
        print(range_info)
        print(center_info)

    if not range_normalized or not center_normalized:
        pc_xyz_tensor = normalize_point_cloud(pc_xyz_tensor, range_normalized, center_normalized, verbose=verbose)


    if if_need_rgb:
        B, N, _ = pc_xyz_tensor.shape

        rgb_tensor = torch.zeros((B, N, 3))
        # rgb_tensor = torch.ones((B, N, 3))
        # rgb_tensor = torch.ones((B, N, 3)) * 100.0

        if if_xyz_rgb:
            pc_tensor = torch.cat((pc_xyz_tensor, rgb_tensor), dim=-1)

            if verbose:
                print("XYZ + RGB")
        else:
            pc_tensor = torch.cat((rgb_tensor, pc_xyz_tensor), dim=-1)

            if verbose:
                print("RGB + XYZ")                
    
    else:
        pc_tensor = pc_xyz_tensor

    if verbose:
        print("Final tensor shape (if combined with RGB):")
        print(pc_tensor.shape)
        
        print("")

    return pc_tensor


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


def encode_text(text, model, if_normalize=False, gpu=0, verbose=False):
    with torch.no_grad():

        text_token = model.tokenizer(text).cuda(gpu, non_blocking=True)

        if verbose:
            print("")
            print("text_token.shape: ")
            print(text_token.shape)

        if len(text_token.shape) < 2:
            text_token = text_token[None, ...]

            if verbose:
                print("text_token.shape after checking: ")
                print(text_token.shape)

        text_embeddings = model.encode_text(text_token).float()

        if if_normalize:
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        if verbose:
            print("text_embeddings.shape")
            print(text_embeddings.shape)
            print("text_embeddings.norm(dim=-1)")
            print(text_embeddings.norm(dim=-1))

            print("")

        return text_embeddings


def encode_image(image, model, if_normalize=False, gpu=0, verbose=False):
    with torch.no_grad():

        image_tensor = model.preprocess(image)

        if verbose:
            # 检查转换后的 tensor 的形状和类型
            print("")
            print("image_tensor.shape after preprocess")
            print(image_tensor.shape)

        img_embedding = model.encode_image(
                                image_tensor.unsqueeze(0).float().cuda(gpu, non_blocking=True)
                                ).float()
        
        if if_normalize:
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
        
        if verbose:
            print("img_embedding.shape")
            print(img_embedding.shape)
            print("img_embedding.norm(dim=-1)")
            print(img_embedding.norm(dim=-1))

            print("")

        return img_embedding
