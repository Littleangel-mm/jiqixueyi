# from tensflow.keras.preprocessing.image import ImagedataGenerator # type: ignore
# import matplotlib.pyplot as plt
# from model import AlexNet_v1
# import tensflow as tf # type: ignore
# import json
# import os

# def main():
#     current_dir = os.getcwd()

#     image_path = os.path.join(current_dir,"flower_data")


#     train_dir = os.path.join(image_path,"train")
#     validation_dir = os.path.join(image_path,"val")

#     assert os.path.exists(train_dir),"cannot find {}".format(train_dir)
#     assert os.path.exists(validation_dir),"cannot find {}".format(validation_dir)

#     if not os.path.exists("save_weights"):
#         os.makedirs("save_weights")







#         enochs = 10

#         train_image_genrator = ImagedataGenerator(rescale = 1/255)

#         train_data_gen = train_image_genrator.flow_from_directory(
#             directory = train_dir,
#             batch_size = batch_size,
#             shuffle = True,
#             class_mode = "categorical"
#         )
#         total_train = train_data_gen.n

#         class_indices = train_data_gen.class_indices
#         inverse_dict = dict((val,key) for key,val in class_indices.items())
#         with open("class_indices.json",'w') as json_file:
#             json_file.write(json_str)

#             val_data_gen = validation_image_generator.flow_from_directory(
#                 directory = train_dir,
#             batch_size = batch_size,
#             shuffle = False,
#             target_size = (im_height,im_width),
#             class_mode = "categorical"

#             )

#             total_val = val_data_gen.n
#             print('fghidjis'.format(total_train,total_val))

#             sample_training_imges,sample_train_labels = next(train_data_gen)


#             model = AlexNet_v1(im_height=image_path,im_width=image_path,num_class=5)
#             model.summary()

#             model.compile(
#                 optimizer = tf.keras.optimizzers(learning_rate=0.0005)
#                 loss = tf.keras.loss.CatgoricalCrossentropy(from_logits=False),
#                 metrics = ['accuracy']
#             )
#             callbacks = [
#                 tf.keras.callbacks.ModelCheckpoint(
#                     filepath = '.save_weights/myAlex.h5',
#                     save_best_only = True,
#                     save_weights_only = True,
#                     monitor = 'val_loss'
#                 )
#             ]
            
#             history = model.fit(
#                 x = train_data_gen,
#                 setps_per_epoch = total_train // batch_size,
#                 epochs = epochs,
#                 validation_data = val_data_gen,
#                 validation_steps = total_val // batch_size
#                 callbacks=callbacks
#             )

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 修正拼写错误
import matplotlib.pyplot as plt
from model import AlexNet_v1
import tensorflow as tf  # 修正拼写错误
import json
import os

def main():
    current_dir = os.getcwd()

    image_path = os.path.join(current_dir, "flower_data")

    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")

    assert os.path.exists(train_dir), "Cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "Cannot find {}".format(validation_dir)

    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    epochs = 10  # 修正拼写错误：enochs -> epochs
    batch_size = 32  # 假设批量大小是 32，你可以根据实际需求调整
    im_height = 224  # 假设图片高度为 224
    im_width = 224  # 假设图片宽度为 224

    # 创建 ImageDataGenerator
    train_image_generator = ImageDataGenerator(rescale=1/255)

    train_data_gen = train_image_generator.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=(im_height, im_width)  # 添加 target_size 参数，确保图片尺寸统一
    )
    total_train = train_data_gen.n

    # 保存类别映射信息
    class_indices = train_data_gen.class_indices
    inverse_dict = {val: key for key, val in class_indices.items()}
    with open("class_indices.json", 'w') as json_file:
        json.dump(inverse_dict, json_file)  # 保存类映射为 JSON 格式

    # 验证集
    validation_image_generator = ImageDataGenerator(rescale=1/255)

    val_data_gen = validation_image_generator.flow_from_directory(
        directory=validation_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(im_height, im_width),
        class_mode="categorical"
    )

    total_val = val_data_gen.n
    print('Total train samples: {}, Total validation samples: {}'.format(total_train, total_val))

    # 获取一个训练样本，检查是否正确
    sample_training_images, sample_train_labels = next(train_data_gen)

    # 创建模型
    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_class=5)  # 假设 5 类
    model.summary()

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # 修正拼写错误：optimizzers -> optimizers
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # 修正拼写错误：loss.CatgoricalCrossentropy -> losses.CategoricalCrossentropy
        metrics=['accuracy']
    )

    # 设置回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='save_weights/myAlex.h5',
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss'
        )
    ]

    # 训练模型
    history = model.fit(
        x=train_data_gen,
        steps_per_epoch=total_train // batch_size,  # 修正拼写错误：setps_per_epoch -> steps_per_epoch
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=total_val // batch_size,
        callbacks=callbacks
    )

    # 可视化训练过程（可选）
    # 画出训练和验证过程中的准确率和损失
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.show()

    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

if __name__ == "__main__":
    main()
