from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from PIL import Image
import numpy as np
import pandas as pds
import matplotlib.pyplot as plt

gobj = ImageDataGenerator(rescale=1. / 255,
                          rotation_range=180,  # ←隨機旋轉 -180~180 度
                          width_shift_range=0.1,  # ←隨機向左或右平移 20% 寬度以內的像素
                          height_shift_range=0.1,  # ←隨機向上或下平移 20% 高度以內的像素
                          brightness_range=(0.7, 1.3), # 亮度
                          zoom_range=0.2,  # ←隨機水平或垂直縮放影像 20% (80%~120%)
                          shear_range=15,  # ←隨機順時針傾斜影像 0~5 度
                          horizontal_flip=True,  # ←隨機水平翻轉影像
                          vertical_flip=True,  # ←隨機垂直翻轉影像
                          fill_mode = 'constant'
                          )

def imageAugmentation(imageList, folder):
    for name in imageList:
        img = []
        img.append(img_to_array(Image.open(folder + '/' + name)))
        plt.imshow(array_to_img(img[0]))
        plt.title(name)
        plt.axis('off')
        plt.show()
        img = np.array(img)

        counter = 0
        for x_data in gobj.flow(img, batch_size=1):
            image = x_data[0]
            image = image.astype('float32')
            image /= 255

            plt.subplot(3, 3, 1 + counter)
            plt.imshow(array_to_img(image))
            plt.axis('off')
            counter += 1
            if counter >= 9:
                break
        plt.title(name)
        plt.show()

file = pds.read_csv("train.csv")
imageList = file.iloc[7: 11, 0]
folder = "C1-P1_Train"
imageAugmentation(imageList, folder)