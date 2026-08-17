
# 每张图片循环调用editcli

# 三个文件夹里的所有图片

#遍历所有图像，对于每张图像，抽取5个caption，每个caption根据edit-cli生成一张对应的图片

#将before图像和5张after图像以及它们对应的caption整理在一张图片中

import json

import numpy as np

import os
import glob
import random
from findpic import extract_file_paths
import subprocess


# def read_lines(file_path, filename):
#     '''
#     file_path: json
#     filename: image file
#     '''
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
    
#     raw_sentences = []
#     for image in data['images']:
#         if image['filename'] == filename:
#             for sentence in image['sentences']:
#                 raw_sentences.append(sentence['raw'])
#             break
#     return raw_sentences

def read_lines(txt_path):
     # 读取文件内容
    with open(txt_path, 'r', encoding='utf-8') as infile:
        sentences = infile.readlines()
    
    # 随机选择 5 个句子
    selected_sentences = random.sample(sentences, 5)
    
    return selected_sentences
    
def generate_output_filename(input_file, edit_option):
    file_name = os.path.basename(input_file)
    base_name = os.path.splitext(file_name)[0]
    # 替换空格为连字符，并生成输出文件名
    edit_formatted = edit_option.replace(" ", "_")
    #直接从json文件里提取的sentence含有.所以这时候png前面不要.
    output_file = f"{base_name}-{edit_formatted}.png"
    return output_file

def run_edit_cli(input_file, edit_option):
    output_file = generate_output_filename(input_file, edit_option)
    input_ = image
    output_ = os.path.join('/root/autodl-tmp/ip2p/newdata_water', output_file)
    print(f"输出文件: {output_}")
    command = ['python', 'edit_cli.py', '--input', input_, '--output', output_, '--edit', edit_option]

    result = subprocess.run(command, capture_output=True, text=True)
    
    # 打印标准输出和标准错误
    if result.returncode != 0:
        print(f"Error: {result.returncode}")
        print("标准错误:", result.stderr)
    else:
        print("成功执行:", result.stdout)

txt_path = '/root/autodl-tmp/ip2p/Levir-MCI-dataset/water.txt'

file_path = '/root/autodl-tmp/ip2p/Levir-MCI-dataset/LevirCCcaptions.json'
folder_path = '/root/autodl-tmp/ip2p/Levir-MCI-dataset/images'  # 替换为你的文件夹路径
images = extract_file_paths(file_path, folder_path)

with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

for image in images:
    #read_lines(file_path,file)的时候用file
    # file = os.path.basename(image)
    for line in read_lines(txt_path):
        run_edit_cli(image, line)
        print(image)
        print(line)


    


