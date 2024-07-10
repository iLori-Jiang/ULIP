with torch.no_grad():
    # generate image feature
    image_feats = []
    target_all = []
    image_pc_similarity_logits_all_images = []
    for idx, image in enumerate(images):
        image_feat = model.encode_image(image[None, ...].cuda())
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        image_feats.append(image_feat)
        for i, (pc, target, target_name) in enumerate(test_loader):
            if idx == 0:
                target_all.extend(target_name)

            pc = pc.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode pc
            pc_features = utils.get_model(model).encode_pc(pc)
            pc_features = pc_features / pc_features.norm(dim=-1, keepdim=True)

            image_pc_similarity_batch = pc_features @ image_feat.t()
            image_pc_similarity.append(image_pc_similarity_batch.squeeze())

            if i % args.print_freq == 0:
                progress.display(i)

        image_pc_similarity_logits = torch.cat(image_pc_similarity, dim=0)

        image_pc_similarity_logits_all_images.append(image_pc_similarity_logits)

    image_pc_similarity_logits_all_images_ensemble = torch.stack(image_pc_similarity_logits_all_images, dim=0).max(dim=0, keepdim=True)[0]
    topk_indices = image_pc_similarity_logits_all_images_ensemble.topk(args.topk)[1].cpu().numpy()
    target_all = np.array(target_all)
    topk_classes = target_all[topk_indices]
    topk_classes_logits = image_pc_similarity_logits[topk_indices]

    progress.synchronize()
    print("topk classes are:")
    print(topk_classes)
    print('topk logits are:')
    print(topk_classes_logits)