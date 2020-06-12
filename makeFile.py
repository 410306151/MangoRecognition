import tensorflow as tf
import pandas as pds
from PIL import Image

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_dataset(fileName, folderName, outputName, size):
    # TFRecord 的檔名
    writer = tf.io.TFRecordWriter(outputName + '.tfrecords')
    for i in range(len(fileName)):
        file = fileName[i]
        folder = folderName[i]

        # 讀取 csv 檔， csv 存放每張圖片的分類等級
        fileTrain = pds.read_csv(file + '.csv', encoding='big5')

        for i in range(fileTrain.shape[0]):
            if fileTrain.iloc[i, 1].upper() == 'A':
                label = 0
            elif fileTrain.iloc[i, 1].upper() == 'B':
                label = 1
            elif fileTrain.iloc[i, 1].upper() == 'C':
                label = 2

            img = Image.open(folder + '/' + fileTrain.iloc[i, 0])
            img = img.resize((size, size))
            img_raw = img.tobytes()

            # 每個 Example 含有 label 、 img_raw 兩個資訊。
            example = tf.train.Example(features = tf.train.Features(feature = {
                "label": _int64_feature(label),
                "img_raw": _bytes_feature(img_raw)
            }))
            # 序列化為字串
            writer.write(example.SerializeToString())
    writer.close()

fileName = ['test_example'] # 'dev' ['label', 'train']
folderName = ['C1-P1_Test'] # 'C1-P1_Dev'  ['sample_image', 'C1-P1_Train']
make_dataset(fileName, folderName, 'validate-test-320', 320)