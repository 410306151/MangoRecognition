import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display

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

def read_and_decode(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)
    apple = list(parsed_image_dataset)

    i = 0
    # 顯示圖片
    for image_features in parsed_image_dataset:
        #print(type(image_features))
        image = tf.io.decode_raw(image_features['img_raw'], tf.uint8)
        image = tf.reshape(image, [256, 256, 3])
        plt.imshow(image)
        plt.title(i)
        plt.show()
        i += 1

def see_without_loop(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    # len(list(dataset)) 在大量資料下會變得很慢，因為先把 iterator 變成 list 來算筆數
    image_batch = parsed_image_dataset.batch(len(list(parsed_image_dataset)))
    # 建構成一個 iterator
    data1 = iter(image_batch).next()
    print(f"\ndata: \n{tf.io.decode_raw(data1['img_raw'][0], tf.uint8).numpy()} \n type: {type(tf.io.decode_raw(data1['img_raw'][0], tf.uint8).numpy())}")

fileName = 'train' # label 、dev 、train
#read_and_decode('train_' + fileName + '.tfrecords')
see_without_loop('train_' + fileName + '.tfrecords')