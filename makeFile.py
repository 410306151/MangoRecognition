import tensorflow as tf
import pandas as pds
import IPython.display as display

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

def make_dataset(fileName, folderName):
    # 讀取 csv 檔， csv 存放每張圖片的分類等級
    fileTrain = pds.read_csv(fileName + '.csv', encoding='big5')
    # TFRecord 的檔名
    writer = tf.io.TFRecordWriter('train.tfrecords')

    for i in range(fileTrain.shape[0]):
        # 將圖片讀成二進制
        img_raw = open(folderName + '/' + fileTrain.iloc[i, 0], 'rb').read()
        if fileTrain.iloc[i, 1].upper() == 'A':
            label = 0
        elif fileTrain.iloc[i, 1].upper() == 'B':
            label = 1
        elif fileTrain.iloc[i, 1].upper() == 'C':
            label = 2
        # 每個 Example 含有 label 、 img_raw 兩個資訊。
        example = tf.train.Example(features = tf.train.Features(feature = {
            "label": _int64_feature(label),
            "img_raw": _bytes_feature(img_raw)
        }))
        # 序列化為字串
        writer.write(example.SerializeToString())
    writer.close()

def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def read_and_decode(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)
    # 顯示圖片
    for image_features in parsed_image_dataset:
        image_raw = image_features['img_raw'].numpy()
        display.display(display.Image(data = image_raw))

fileName = 'label'
folderName = 'sample_image'
#make_dataset(fileName, folderName)
read_and_decode('train.tfrecords')
