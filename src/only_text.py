'''
Description: 
Author: 唐健峰
Date: 2023-12-28 10:26:32
LastEditors: ${author}
LastEditTime: 2024-01-29 00:46:21
'''
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, concatenate,Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pre import load_images_and_texts
# 加载图像和文本数据
images_and_texts = load_images_and_texts("resources/实验五数据/data")

 # 图像处理
def preprocess_image(img, target_size=(224, 224), normalization=True):
        # 将图像调整为目标大小
        img = img.resize(target_size)
        
        # 将图像转为 NumPy 数组
        img_array = img_to_array(img)

        # 可选：归一化，将像素值缩放到 [0, 1] 范围
        if normalization:
            img_array /= 255.0

        # 返回预处理后的图像数组
        return img_array

def get_model(images_and_texts):

    # 从CSV文件中读取标签信息
    labels = [None] * 5129
    file_path = 'resources/实验五数据/train.txt'
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # 跳过CSV文件的标题行
        next(csv_reader, None)
        
        for row in csv_reader:
            if len(row) == 2:
                guid = int(row[0])
                tag = row[1]
                
                # 将标签信息存储到labels列表中
                labels[guid - 1] = tag

    # 获取非 None 的标签及其对应的 images_and_texts
    valid_labels_and_data = [(label, image_and_text) for label, image_and_text in zip(labels, images_and_texts) if label is not None]

    # 将原始的 labels 和 images_and_texts 替换为非 None 的数据
    labels, images_and_texts = zip(*valid_labels_and_data)


    # 将标签编码为数字
    label_mapping = {'negative': 0, 'positive': 2, 'neutral': 1}
    encoded_labels = [label_mapping[label] for label in labels]

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(images_and_texts, encoded_labels, test_size=0.2, random_state=42)


    X_train_images = np.array([preprocess_image(img) for img, _ in X_train])
    X_val_images = np.array([preprocess_image(img) for img, _ in X_val])

    # 文本处理
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text for _, text in X_train])
    X_train_texts = tokenizer.texts_to_sequences([text for _, text in X_train])
    X_val_texts = tokenizer.texts_to_sequences([text for _, text in X_val])

    max_length_of_texts = max(len(text) for _, text in images_and_texts if text is not None)
    max_length_of_texts = 400

    # 使用 pad_sequences 设置 maxlen，确保文本序列的长度为期望的长度
    X_train_texts = pad_sequences(X_train_texts, maxlen=max_length_of_texts)
    X_val_texts = pad_sequences(X_val_texts, maxlen=max_length_of_texts)

    # 获取词汇表的大小
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 10

    # 定义模型
    # 假设图像尺寸为 (224, 224) ，通道数为 3（RGB）
    height, width, channels = 224, 224, 3
    image_input = Input(shape=(height, width, channels))  # 替换为你的图像尺寸
    text_input = Input(shape=(max_length_of_texts,))

    # 图像处理部分
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    # 添加Dropout层
    x = Dropout(0.5)(x)
    # Flatten层
    x = Flatten()(x)

    # 文本处理部分
    y = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    y = LSTM(64, return_sequences=True)(y)  # 将return_sequences参数设置为True
    y = SpatialDropout1D(0.5)(y)
    y = Flatten()(y)

    # 合并图像和文本处理结果
    combined = concatenate([y])

    # 输出层
    output = Dense(3, activation='sigmoid')(combined)

    # 创建模型
    model = Model(inputs=[text_input], outputs=output)

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 添加EarlyStopping回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 训练模型
    model.fit([X_train_texts], np.array(y_train), 
            epochs=4, batch_size=16, 
            validation_data=([X_val_texts], np.array(y_val)),
            callbacks=[early_stopping])

    model.save('src/model.h5')
    return model, tokenizer

model, tokenizer = get_model(images_and_texts)
