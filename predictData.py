import tensorflow as tf
import numpy as np
import pandas as pds

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

def getTFRecordsData(filename):
    # 讀入 TFRecords 檔
    file = tf.data.TFRecordDataset(filename)

    # Parse Data
    parsed_image_dataset = file.map(_parse_function)

    # len(list(dataset)) 在大量資料下會變得很慢，因為先把 iterator 變成 list 來算筆數
    image_batch = parsed_image_dataset.batch(len(list(parsed_image_dataset)))

    return image_batch

def loadModel(modelFile):
    return tf.keras.models.load_model(modelFile)

def predictLabel(modelName, fileName):
    size = 320
    # 讀入 TFRecords 檔
    image_batch = getTFRecordsData(fileName)

    # 建構成一個 iterator
    data = iter(image_batch).next()

    image = []
    for i in range(len(data['label'])):
        temp = tf.io.decode_raw(data['img_raw'][i], tf.uint8)
        image.append(tf.reshape(temp, [size, size, 3]).numpy())
    image = np.array(image) / 255
    # 模型要求要 4 dimensions ，所以在這裡做 reshape 變成四維
    image = image.reshape(image.shape[0], size, size, 3)
    model = loadModel(modelName)

    return np.argmax(model.predict(image), axis=1)

def convertToLebel(output):
    labelPredict = []

    for i in range(len(output)):
        if output[i] == 0:
            labelPredict.append("A")
        elif output[i] == 1:
            labelPredict.append("B")
        elif output[i] == 2:
            labelPredict.append("C")

    return labelPredict

def outputCSV(imageID, labelPredict, outputName):
    # Output CSV file
    csv = pds.DataFrame()
    csv['image_id'] = imageID
    csv['label'] = labelPredict
    csv.to_csv(outputName)

# 讀取 Model 並預測資料
output = predictLabel('2_vgg19_weights.20-0.82.h5', 'validate-test-320.tfrecords')
# Ouput 的 label 轉為 A, B, C 等級
labelPredict = convertToLebel(output)
# 讀 CSV 檔拿 Image ID
valFile = pds.read_csv("test_example.csv")
# 輸出成 CSV 檔
outputCSV(valFile.iloc[:, 0], labelPredict, 'answer.csv')
