import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical

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
    input_shape = (256, 256, 3)
    model = Sequential()

    model.add(Conv2D(64, kernel_size = (3, 3), input_shape = input_shape, padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding = 'same', ))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model

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
    # Checkpoint location
    checkpoint_path = "training_1/MangoIsGood-{epoch:04d}.ckpt"
    #checkpoint_path = "/content/gdrive/My Drive/Mango/training_1/MangoIsGood-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # 每 10 個 epochs 存一次紀錄點
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                                     save_weights_only = True,
                                                     verbose = 1,
                                                     #save_freq = 5,
                                                     period = 10)

    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    #file = tf.data.TFRecordDataset("/content/gdrive/My Drive/Mango/" + filename)

    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    # len(list(dataset)) 在大量資料下會變得很慢，因為先把 iterator 變成 list 來算筆數
    image_batch = parsed_image_dataset.batch(len(list(parsed_image_dataset)))

    # 建構成一個 iterator
    data = iter(image_batch).next()
    #print(f"\ndata: \n{tf.io.decode_raw(data['img_raw'][0], tf.uint8).numpy()} \n type: {type(tf.io.decode_raw(data['img_raw'][0], tf.uint8).numpy())}")

    # 建構模型
    model = build_model()
    #model.summary()
    # 如果有 checkpoint 讀取最新的 checkpoint
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is not None:
        model.load_weights(latest)

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
    model.fit(train, label, batch_size = 64, epochs = 60, verbose = 2, callbacks = [cp_callback])

fileName = 'data' # label 、dev 、train
training('train_' + fileName + '.tfrecords')
#show_image('train_' + fileName + '.tfrecords')