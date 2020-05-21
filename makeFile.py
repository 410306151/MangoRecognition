import tensorflow as tf
import pandas as pds
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image

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

def make_dataset(fileName, folderName):
    doResize = False
    # 讀取 csv 檔， csv 存放每張圖片的分類等級
    fileTrain = pds.read_csv(fileName + '.csv', encoding='big5')
    # TFRecord 的檔名
    writer = tf.io.TFRecordWriter('train.tfrecords')

    for i in range(fileTrain.shape[0]):
        # 將圖片讀成二進制
        if doResize :
            img = Image.open(folderName + '/' + fileTrain.iloc[i, 0])
            img = img.resize((256, 256))
            img_raw = img.tobytes()
        else :
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

def read_and_decode(filename):
    doResize = False
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    i = 0
    # 顯示圖片
    for image_features in parsed_image_dataset:
        #print(type(image_features))
        if doResize :
            image = tf.io.decode_raw(image_features['img_raw'], tf.uint8)
            image = tf.reshape(image, [256, 256, 3])
            plt.imshow(image)
            plt.title(i)
            plt.show()
            i += 1
        else:
            image = tf.io.decode_raw(image_features['img_raw'], tf.uint8)
            #display.display(display.Image(data = image))

def see_without_loop(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)
    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    # len(list(dataset)) 在大量資料下會變得很慢，因為先把 iterator 變成 list 來算筆數
    #image_batch = parsed_image_dataset.batch(len(list(parsed_image_dataset)))
    # 建構成一個 iterator
    #data1 = iter(image_batch).next()
    #print(data1['label'][60:])
    
    # 建構成一個 list
    data2 = list(parsed_image_dataset)
    print(data2[0]['label'])

fileName = 'label' # label 、dev 、train
folderName = 'sample_image' # sample_image 、C1-P1_Dev 、 C1-P1_Train
#make_dataset(fileName, folderName)
#read_and_decode('train.tfrecords')
see_without_loop('train.tfrecords')