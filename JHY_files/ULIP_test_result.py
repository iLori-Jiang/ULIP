# ------------------------v1

logits_img_A = img_A_embedding @ class_embeddings.t()
prob_img_A = F.softmax(logits_img_A, dim=-1)

print(logits_img_A)
print(prob_img_A)

# tensor([[0.4366, 0.2976, 0.2654]], device='cuda:0')
# tensor([[0.3686, 0.3208, 0.3106]], device='cuda:0')

logits_img_B = img_B_embedding @ class_embeddings.t()
prob_img_B = F.softmax(logits_img_B, dim=-1)

print(logits_img_B)
print(prob_img_B)

# tensor([[0.4238, 0.2766, 0.2579]], device='cuda:0')
# tensor([[0.3690, 0.3185, 0.3126]], device='cuda:0')

sim_img_A_B = img_A_embedding @ img_B_embedding.t()
print(sim_img_A_B)
# tensor([[0.9710]], device='cuda:0')

logits_per_pc = pc_features @ class_embeddings.t()
print(logits_per_pc)

# tensor([[0.1032, 0.0661, 0.0878]], device='cuda:0')
# NOTE: shouldn't use the point cloud with the class of image

probabilities = F.softmax(logits_per_pc, dim=-1)
print(probabilities)

# tensor([[0.3392, 0.3268, 0.3340]], device='cuda:0')

sim_pc_A = pc_features @ img_A_embedding.t()
sim_pc_B = pc_features @ img_B_embedding.t()

print(sim_pc_A)
print(sim_pc_B)
# tensor([[0.1139]], device='cuda:0')
# tensor([[0.1251]], device='cuda:0')

print(pc_features @ text_B_embeddings.t())
print(img_A_embedding @ text_B_embeddings.t())
print(img_B_embedding @ text_B_embeddings.t())
# tensor([[0.1476]], device='cuda:0')
# tensor([[0.3808]], device='cuda:0')
# tensor([[0.3968]], device='cuda:0')

logits_per_pc_mn40 = pc_features_mn40 @ class_embeddings.t()
print(logits_per_pc_mn40)
# tensor([[0.1032, 0.0661, 0.0879]], device='cuda:0')

prob_mn40 = F.softmax(logits_per_pc_mn40, dim=-1)
print(prob_mn40)
# tensor([[0.3392, 0.3268, 0.3340]], device='cuda:0')

print(pc_features_mn40 @ pc_features.t())
print(pc_features_mn40 @ img_A_embedding.t())
print(pc_features_mn40 @ img_B_embedding.t())
print(pc_features_mn40 @ text_B_embeddings.t())
# tensor([[1.0000]], device='cuda:0')       # NOTE: shouldn't happen, maybe the xyz and rgb should switch?
# tensor([[0.1140]], device='cuda:0')
# tensor([[0.1252]], device='cuda:0')
# tensor([[0.1473]], device='cuda:0')
