import os
import re
import torch
import clip
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
# model_path = "./ViT-B-32.pt"
# model, preprocess = clip.load(model_path, device=device, jit=False)  
model, preprocess = clip.load("ViT-B/32")
model.eval()
# 路径配置（请根据实际情况修改）
generate_folder = "/root/autodl-tmp/ip2p/newdata_water"  # 生成图像文件夹
input_folder = "/root/autodl-tmp/ip2p/before_water"        # 原始图像文件夹

cosine_scores = []
pattern = re.compile(r'^(train_\d+)\.(png|jpg)$', re.IGNORECASE)  # 匹配基础文件名

for filename in os.listdir(generate_folder):
    if not filename.lower().endswith(('.png')):
        continue
    
    # 提取对应的原始图像文件名
    match = pattern.match(filename)
    if not match:
        print(f"Skipping {filename} (name pattern mismatch)")
        continue
    
    base_name = f"{match.group(1)}.png"  # 原始图像扩展名设为.png
    original_path = os.path.join(input_folder, base_name)
    
    if not os.path.exists(original_path):
        print(f"Original image {original_path} not found")
        continue

    try:
        gen_image = preprocess(Image.open(os.path.join(generate_folder, filename)))
        gen_image = gen_image.unsqueeze(0).to(device)
        
        real_image = preprocess(Image.open(original_path))
        real_image = real_image.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading images: {e}")
        continue
    with torch.no_grad():
        gen_features = model.encode_image(gen_image).cpu().numpy()
        real_features = model.encode_image(real_image).cpu().numpy()
    
    similarity = cosine_similarity(gen_features, real_features)
    cosine_scores.append(similarity[0][0])

if cosine_scores:
    avg_similarity = np.mean(cosine_scores)
    print(f"CLIP-I Score: {avg_similarity:.4f}")
else:
    print("No valid image pairs processed")