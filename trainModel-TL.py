# Train model with transfor learning
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

# Create a description of the features.
feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'img_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def build_model():
    vgg16 = VGG16(include_top=False,
               weights='imagenet',
               input_shape=(256, 256, 3))

    unfreeze = ['block5_conv1', 'block5_conv2', 'block5_conv3'] # 最後 3 層的名稱

    for layer in vgg16.layers:
        if layer.name in unfreeze:
            layer.trainable = True  # 最後 3 層解凍
        else:
            layer.trainable = False # 其他凍結權重

    model = Sequential()
    model.add(vgg16)    # 將 vgg16 做為一層
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
               metrics=['acc'])

    return model

def getTFRecordsData(filename, googleDrive = 0):
    if googleDrive == 1:
        googleDrivePath = "/content/gdrive/My Drive/Mango/"
    else:
        googleDrivePath = ""

    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(googleDrivePath + filename)

    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    # len(list(dataset)) 在大量資料下會變得很慢，因為先把 iterator 變成 list 來算筆數
    image_batch = parsed_image_dataset.batch(len(list(parsed_image_dataset)))

    return image_batch

def show_image(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    i = 0
    # 顯示圖片
    for image_features in parsed_image_dataset:
        image = tf.io.decode_raw(image_features['img_raw'], tf.uint8)
        image = tf.reshape(image, [256, 256, 3])
        plt.imshow(image)
        plt.title(i)
        plt.show()
        i += 1

def training(filename):
    googleDrive = 0
    if googleDrive == 1:
        googleDrivePath = "/content/gdrive/My Drive/Mango/"
    else:
        googleDrivePath = ""
    # Checkpoint location
    checkpoint_path = googleDrivePath + "training_1/MangoIsGood-data-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # 每 10 個 epochs 存一次紀錄點
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                     save_weights_only = True,
                                                     verbose = 1,
                                                     #save_freq = 5,
                                                     period = 10)

    # 讀入 TFRecords 檔
    image_batch = getTFRecordsData(filename, googleDrive)

    # 建構成一個 iterator
    data = iter(image_batch).next()

    # 建構模型
    model = build_model()
    # 如果有 checkpoint 讀取最新的 checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    #if latest is not None:
        #model.load_weights(latest)
    #model.summary()

    # 取資料後訓練
    train = []
    label = []
    for i in range(len(data['label'])):
        temp = tf.io.decode_raw(data['img_raw'][i], tf.uint8)
        train.append(tf.reshape(temp, [256, 256, 3]).numpy())
        label.append(data['label'][i])
    train = np.array(train)
    label = np.array(label)
    label = to_categorical(label, 3)
    # 模型要求要 4 dimensions ，所以在這裡做 reshape 變成四維
    train = train.reshape(train.shape[0], 256, 256, 3)
    model.fit(train, label, batch_size = 64, epochs = 90, verbose = 2, callbacks = [cp_callback])
    model.save(googleDrivePath)

fileName = 'data'
training('train_' + fileName + '.tfrecords')
#show_image('train_' + fileName + '.tfrecords')