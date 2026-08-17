import json
import os

# 定义输入 JSON 文件路径和输出文件夹路径
json_file_path = '/root/autodl-tmp/instruct/Levir-MCI-dataset/LevirCCcaptions.json'
output_folder = '/root/autodl-tmp/instruct/Levir-MCI-dataset/captions'

# 创建输出文件夹（如果不存在）
os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)

# 读取 JSON 文件
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

print(type(data))  # 应该是 <class 'list'>
print("data keys:", data.keys())

# # 遍历每个样本
# for item in data:
#     filepath = item['filepath']
#     filename = os.path.splitext(item['filename'])
#     sentences = item['sentences']

#     # 确定输出文件夹
#     output_dir = os.path.join(output_folder, filepath)

#     # 创建 TXT 文件名
#     txt_file_name = os.path.join(output_dir, f"{filename}.txt")

#     # 写入 sentences 中的 raw 内容到 TXT 文件
#     with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
#         for sentence in sentences:
#             txt_file.write(sentence['raw'].strip() + '\n')

# print("所有样本已成功保存到对应的 TXT 文件中。")

# 遍历 images 列表
if 'images' in data:
    for item in data['images']:  # 访问 images 列表中的每个样本
        filepath = item['filepath']
        filename = os.path.splitext(item['filename'])[0]
        sentences = item['sentences']

        # 确定输出文件夹
        output_dir = os.path.join(output_folder, filepath)

        # 创建 TXT 文件名
        txt_file_name = os.path.join(output_dir, f"{filename}.txt")

        # 写入 sentences 中的 raw 内容到 TXT 文件
        with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
            for sentence in sentences:
                txt_file.write(sentence['raw'].strip() + '\n')
else:
    print("No 'images' key found in data.")

print("处理完成。")