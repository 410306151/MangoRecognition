import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
import IPython.display as display

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def make_dataset(fileName, folderName):
    fileTrain = pds.read_csv(fileName + '.csv', encoding='big5')
    writer = tf.io.TFRecordWriter('train.tfrecords') #輸出成tfrecord檔案

    for i in range(fileTrain.shape[0]):
        img_path = folderName + '/' + fileTrain.iloc[i, 0] #每個圖片的地址
        #img = Image.open(img_path)
        #img = img.resize((208, 208))
        #img_raw = img.tobytes()  #將圖片轉化為二進位制格式
        img_raw = open(folderName + '/' + fileTrain.iloc[i, 0], 'rb').read()
        if fileTrain.iloc[i, 1].upper() == 'A':
            label = 0
        elif fileTrain.iloc[i, 1].upper() == 'B':
            label = 1
        elif fileTrain.iloc[i, 1].upper() == 'C':
            label = 2
        example = tf.train.Example(features = tf.train.Features(feature = {
            "label": _int64_feature(label),
            "img_raw": _bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())  #序列化為字串
    writer.close()

# Create a description of the features.
feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'img_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

def read_and_decode(filename, batch_size): # read train.tfrecords
    filename_queue = tf.data.TFRecordDataset(filename)# create a queue
    parsed_image_dataset = filename_queue.map(_parse_function)
    for image_features in parsed_image_dataset:
        image_raw = image_features['img_raw'].numpy()
        display.display(display.Image(data = image_raw))

fileName = 'label'
folderName = 'sample_image'
#make_dataset(fileName, folderName)

read_and_decode('train.tfrecords', 32)

#show_image()