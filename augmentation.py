import os
#os.chdir("My Drive/")
#os.chdir("..")
#print(os.listdir())

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
#import util7 as u

input_shape = (300,300,3)
target_size = input_shape[:2]

gobj = ImageDataGenerator(rescale=1. / 255,
                          rotation_range=180,  # ←隨機旋轉 -180~180 度 (會產生殘影不能用)
                          width_shift_range=0.1,  # ←隨機向左或右平移 20% 寬度以內的像素
                          height_shift_range=0.1,  # ←隨機向上或下平移 20% 高度以內的像素
                          brightness_range=(0.7, 1.3), # (可用但需調整)
                          zoom_range=0.1,  # ←隨機水平或垂直縮放影像 20% (80%~120%) (可用但不能太大)
                          shear_range=0,  # ←隨機順時針傾斜影像 0~5 度 (不好)
                          horizontal_flip=True,  # ←隨機水平翻轉影像 (可用)
                          vertical_flip=True,  # ←隨機垂直翻轉影像 (可用)
                          fill_mode = 'constant'                          )

trn_gen = gobj.flow_from_directory(
    'dev',
    target_size=target_size,
    #batch_size=50,  # total 5600
    class_mode='categorical')

batches = 0
for x_data, x_label in trn_gen:
    for i in range(len(trn_gen)): # 找這個 batch 中有幾筆資料
        plt.imshow(x_data[i])
        plt.title(batches)
        plt.show()
    if batches >= 30:
        break
    batches += 1
#plt.imshow(trn_gen[0][0][0])
#plt.show()