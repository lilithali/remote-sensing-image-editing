from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

import os
import cv2
import glob
import random


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.split = split
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        
        self.image_file_A = os.path.join(self.path, "images", split, "A")
        self.image_file_B = os.path.join(self.path, "images", split, "B")
        
        self.image_paths_A = glob.glob(os.path.join(self.image_file_A, "*.png"))   #返回 self.image_file_A 目录下所有以 .png 结尾的文件的完整路径。
        self.image_paths_B = glob.glob(os.path.join(self.image_file_B, "*.png"))
        self.caption_folder = os.path.join(self.path, "captions", split)
        

            
    def __len__(self) -> int:
        return len(self.image_paths_A)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        
        example = {}
        image_path_A = self.image_paths_A[idx]
        image_path_B = self.image_paths_B[idx]
        
        image_0 = Image.open(image_path_A)
        image_1 = Image.open(image_path_B)
        # seed = seeds[torch.randint(0, len(seeds), ()).item()]
        
        # with open(propt_dir.joinpath("prompt.json")) as fp:
        # prompt = datafiles["sentence"]

        # image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.png"))
        # image_1 = Image.open(propt_dir.joinpath(f"{seed}_1.png"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
       

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")
        # print("333333333333",type(image_1))
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)
        
        filename = os.path.splitext(os.path.basename(image_path_A))[0]
        # os.path.basename(...) 函数用于获取路径中的基本文件名部分
        # os.path.splitext(...) 函数用于将文件名分割成文件名和扩展名两部分
        # 这个索引操作提取元组的第一个元素，即文件名部分，不包含扩展名
        txt_path = os.path.join(self.caption_folder, filename + ".txt")
        with open(txt_path, 'r') as f:  #f 是一个文件对象，它代表打开的文件
            lines = f.readlines()   #readlines() 方法用于读取文件中的所有行，并将它们作为一个列表返回
        captions = [line[:-2] for line in lines]  #line[:-2] 是切片操作，表示从字符串的开头到倒数第二个字符的所有字符
        prompt = captions[0]

        
        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))



# 设置数据集路径
data_path = "/root/autodl-tmp/ip2p/Levir-MCI-dataset"  # 替换为您的实际数据集路径

# 创建 EditDataset 实例
dataset = EditDataset(
    path=data_path,
    split="train",  # 可以是 "train", "val", 或 "test"
    min_resize_res=256,
    max_resize_res=256,
    crop_res=256,
    flip_prob=0.5  # 根据需要设置翻转概率
)

# 打印图像数量（可选）
print(f"Loaded dataset with {len(dataset.image_paths_A)} images in A and {len(dataset.image_paths_B)} images in B.")







# class EditDatasetEval(Dataset):
#     def __init__(
#         self,
#         path: str,
#         split: str = "train",
#         splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
#         res: int = 256,
#     ):
#         assert split in ("train", "val", "test")
#         assert sum(splits) == 1
#         self.path = path
#         self.res = res

#         with open(Path(self.path, "seeds.json")) as f:
#             self.seeds = json.load(f)

#         split_0, split_1 = {
#             "train": (0.0, splits[0]),
#             "val": (splits[0], splits[0] + splits[1]),
#             "test": (splits[0] + splits[1], 1.0),
#         }[split]

#         idx_0 = math.floor(split_0 * len(self.seeds))
#         idx_1 = math.floor(split_1 * len(self.seeds))
#         self.seeds = self.seeds[idx_0:idx_1]

#     def __len__(self) -> int:
#         return len(self.seeds)

#     def __getitem__(self, i: int) -> dict[str, Any]:
#         name, seeds = self.seeds[i]
#         propt_dir = Path(self.path, name)
#         seed = seeds[torch.randint(0, len(seeds), ()).item()]
#         with open(propt_dir.joinpath("prompt.json")) as fp:
#             prompt = json.load(fp)
#             edit = prompt["edit"]
#             input_prompt = prompt["input"]
#             output_prompt = prompt["output"]

#         image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

#         reize_res = torch.randint(self.res, self.res + 1, ()).item()
#         image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

#         image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

#         return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)





