'''
Description: 
Author: 唐健峰
Date: 2024-01-27 19:12:15
LastEditors: ${author}
LastEditTime: 2024-01-27 19:38:43
'''
from PIL import Image
import os

def load_images_and_texts(folder_path):
    images_and_texts = [None] * 5129
    
    # 获取文件夹中所有文件的路径
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
    
    for file_path in file_paths:
        # 仅处理以".jpg"和".txt"结尾的文件
        if file_path.endswith(".jpg"):
            img = Image.open(file_path)
            
            # 获取文件名中的数字部分
            file_number = int(os.path.splitext(os.path.basename(file_path))[0])
            
            # 寻找相应的文本文件
            txt_file_path = os.path.join(folder_path, f"{file_number}.txt")
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r', encoding='latin-1') as txt_file:
                    text = txt_file.read()
                    images_and_texts[file_number - 1] = (img, text)
    
    return images_and_texts


