# from tensflow.keras import layers,models # type: ignore

# def AlexNet_v1(im_height=224,im_width=224,num_class=1000):
#     input_image = layers.Input(shape=(im_height,im_width,3),dtype = "float32")
#     x = layers.ZeroPadding2D(((1,2),(1,2)))(input_image)
#     x = layers.Conv2D(48,kernel_size=11,strides=4,activation='relu')(x)
#     x = layers.Maxpool2D(pool_size=3,strides = 2)(x)
#     x = layers.Conv2D(128,kernel_size=5,padding='same',activation="relu")(x)

#     x = layers.Maxpool2D(pool_size = 3,strides = 2)(2)


from tensorflow.keras import layers, models  # 修正拼写错误：tensflow -> tensorflow

def AlexNet_v1(im_height=224, im_width=224, num_class=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")  # 输入层

    # 第一层卷积 + 最大池化
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # 添加 Zero Padding
    x = layers.Conv2D(48, kernel_size=11, strides=4, activation='relu')(x)  # 卷积层
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 最大池化层

    # 第二层卷积 + 最大池化
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)  # 卷积层
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 最大池化层

    # 第三层卷积
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # 卷积层

    # 第四层卷积
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # 卷积层

    # 第五层卷积 + 最大池化
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)  # 卷积层
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 最大池化层

    # 扁平化层
    x = layers.Flatten()(x)

    # 全连接层1
    x = layers.Dense(2048, activation='relu')(x)  # 全连接层
    x = layers.Dropout(0.5)(x)  # Dropout层，避免过拟合

    # 全连接层2
    x = layers.Dense(2048, activation='relu')(x)  # 全连接层
    x = layers.Dropout(0.5)(x)  # Dropout层，避免过拟合

    # 输出层
    output = layers.Dense(num_class, activation='softmax')(x)  # 分类层

    # 构建模型
    model = models.Model(inputs=input_image, outputs=output)

    return model
