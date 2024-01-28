'''
Description: 
Author: 唐健峰
Date: 2023-12-28 10:26:32
LastEditors: ${author}
LastEditTime: 2024-01-29 02:39:04
'''
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv1D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, concatenate,Dropout, SpatialDropout1D, Attention, GlobalMaxPooling2D, GlobalMaxPooling1D, Multiply, RepeatVector
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
    # 图像处理部分
    height, width, channels = 224, 224, 3
    image_input = Input(shape=(height, width, channels))
    text_input = Input(shape=(max_length_of_texts,))

    # 图像处理部分
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    # 文本处理部分
    y = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(text_input)
    y = LSTM(128, return_sequences=True)(y)
    y = SpatialDropout1D(0.5)(y)
    y = Flatten()(y)

    # 合并图像和文本处理结果
    combined = concatenate([x, y])

    # 输出层
    output = Dense(3, activation='softmax')(combined)

    # 创建模型
    model = Model(inputs=[image_input, text_input], outputs=output)
    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 添加EarlyStopping回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 训练模型
    model.fit([X_train_images, X_train_texts], np.array(y_train), 
            epochs=4, batch_size=16, 
            validation_data=([X_val_images, X_val_texts], np.array(y_val)),
            callbacks=[early_stopping])

    model.save('src/model.h5')
    return model, tokenizer

model, tokenizer = get_model(images_and_texts)

# 加载未标记的数据
file_path = 'resources/实验五数据/test_without_label.txt'
unlabeled_data = []
with open(file_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)
    
    # 跳过CSV文件的标题行
    next(csv_reader, None)
    
    for row in csv_reader:
        if len(row) == 2:
            guid = int(row[0])
            unlabeled_data.append(guid)  # 注意这里只存储guid而不是元组

# 加载模型
model = load_model('src/model.h5')

# 获取未标记数据的图像和文本
unlabeled_images_and_texts = [images_and_texts[guid - 1] for guid in unlabeled_data]

# 图像预处理
processed_images = np.array([preprocess_image(img) for img, _ in unlabeled_images_and_texts])

# 文本预处理
texts_to_predict = [text for _, text in unlabeled_images_and_texts]
processed_texts = tokenizer.texts_to_sequences(texts_to_predict)
processed_texts = pad_sequences(processed_texts, maxlen=400)

# 进行一次性的预测
predictions = model.predict([processed_images, processed_texts])

# 获取所有预测结果
predicted_labels = [(guid, np.argmax(prediction)) for guid, prediction in zip(unlabeled_data, predictions)]

# 输出预测结果到文件
label_mapping = {'negative': 0, 'positive': 2, 'neutral': 1}
output_file_path = 'resources/实验五数据/test_without_label_predictions.txt'
with open(output_file_path, 'w', newline='', encoding='utf-8') as output_file:
    output_file.write('guid,tag\n')
    for guid, predicted_label in predicted_labels:
        tag = [key for key, value in label_mapping.items() if value == predicted_label][0]
        output_file.write(f"{guid},{tag}\n")

print(f"Predictions saved to {output_file_path}")
