from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import re
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CLIPModel.from_pretrained("/root/autodl-tmp/ip2p/stable_diffusion/ldm/modules/encoders/openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/ip2p/stable_diffusion/ldm/modules/encoders/openai/clip-vit-large-patch14")

# def extract_prompt_from_filename(filename):

#     name_without_ext = os.path.splitext(filename)[0]

#     # parts = name_without_ext.split('_')
#     parts = re.split(r'[_-]', name_without_ext)
#     # 前两部分是前缀和编号，从第三部分开始是prompt
#     if len(parts) >= 3:
#         return ' '.join(parts[2:])
#     return ''

def extract_prompt_from_filename(file_path, filename):
    '''
    file_path: json
    filename: image file
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for image in data['images']:
        if image['filename'] == filename:
            if image['sentences']:  # 确保句子列表不为空
                return image['sentences'][0]['raw']  # 返回第一个句子的 raw
    return None  # 如果没有找到，返回 None
    
def get_all_images(folder_path):

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png'))]

def get_clip_score(image_paths, texts):
    """计算一批图片和对应文本的CLIP分数"""
    images = [Image.open(p) for p in image_paths]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = outputs.logits_per_image.diag()
    return scores.cpu().numpy()

def calculate_clip_scores(images_folder_path, file_path, batch_size=1):

    image_paths = get_all_images(images_folder_path)
    print(f"Found {len(image_paths)} images")
    
    text_prompts = []
    for path in image_paths:
        filename = os.path.basename(path)
        prompt = extract_prompt_from_filename(file_path,filename)
        print('prompt',prompt)
        text_prompts.append(prompt)
    
    all_scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), 
                desc="Calculating CLIP Scores"):
        batch_paths = image_paths[i:i+batch_size]
        batch_texts = text_prompts[i:i+batch_size]
        
        if not batch_texts:  # 跳过空批次
            continue
            
        batch_scores = get_clip_score(batch_paths, batch_texts)
        all_scores.extend(batch_scores)
    
    return np.array(all_scores)

if __name__ == "__main__":
    images_folder_path = '/root/autodl-tmp/ip2p/Levir-MCI-dataset/images/train/B1'
    file_path = '/root/autodl-tmp/ip2p/Levir-MCI-dataset/LevirCCcaptions1.json'
    scores = calculate_clip_scores(images_folder_path, file_path)

    print("\nIndividual Scores:")
    for path, score in zip(get_all_images("/root/autodl-tmp/ip2p/Levir-MCI-dataset/images/train/B1"), scores):
        print(f"{os.path.basename(path)}: {score:.4f}")
    
    print(f"\nAverage CLIP Score: {np.mean(scores):.4f}")
    print(f"Std CLIP Score: {np.std(scores):.4f}")