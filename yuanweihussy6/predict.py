# from tensflow.keras import layers,models # type: ignore

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models  # 修正拼写错误：tensflow -> tensorflow
import json
import matplotlib.pyplot as plt

def AlexNet_v1(im_height=224, im_width=224, num_class=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_class, activation='softmax')(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model

# 加载训练好的模型
def load_trained_model(model_path='save_weights/myAlex.h5'):
    # 加载模型权重
    model = AlexNet_v1(im_height=224, im_width=224, num_class=5)  # 假设模型有5个类
    model.load_weights(model_path)
    print("Model loaded successfully.")
    return model

# 图像预处理函数
def prepare_image(image_path, im_height=224, im_width=224):
    img = image.load_img(image_path, target_size=(im_height, im_width))
    img_array = image.img_to_array(img)  # 转换为数组
    img_array = np.expand_dims(img_array, axis=0)  # 扩展维度以适应模型输入
    img_array = img_array / 255.0  # 归一化处理
    return img_array

# 对新图像进行预测
def predict_image(model, image_path):
    img_array = prepare_image(image_path)
    predictions = model.predict(img_array)  # 模型预测
    predicted_class = np.argmax(predictions, axis=1)  # 获取最大概率的类索引
    return predicted_class, predictions

# 读取类别映射文件
def load_class_indices(file_path='class_indices.json'):
    with open(file_path, 'r') as json_file:
        class_indices = json.load(json_file)
    inverse_dict = {v: k for k, v in class_indices.items()}  # 将索引映射到类名
    return inverse_dict

# 主函数
def main():
    model_path = 'save_weights/myAlex.h5'
    image_path = 'path_to_your_image.jpg'  # 要预测的图片路径

    # 加载训练好的模型
    model = load_trained_model(model_path)

    # 进行预测
    predicted_class, predictions = predict_image(model, image_path)

    # 加载类别映射
    class_indices = load_class_indices()

    # 输出预测结果
    print(f"Predicted class index: {predicted_class[0]}")
    print(f"Predicted class name: {class_indices[predicted_class[0]]}")
    print(f"Prediction probabilities: {predictions}")

    # 可视化图像和预测结果
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {class_indices[predicted_class[0]]}")
    plt.show()

if __name__ == "__main__":
    main()
