from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pds


def loadImage(fileName, folderName, showImage = False):
    imgArray = []

    # 讀取圖片
    for i in range(fileTrain.shape[0]):
        #img = image.load_img('D-Plant2_0610_3.jpg', target_size=(224, 224))
        img2 = image.load_img(folderName + '/' + fileTrain.iloc[i, 0])
        x = image.img_to_array(img2)
        imgArray.append(img2)
        if showImage:
            plt.imshow(x/255.)
            plt.show()
    return imgArray

def main():
    imageString = []
    fileName = 'label'
    folderName = 'sample_image'
    fileTrain = pds.read_csv(fileName + '.csv', encoding='big5')

    for i in range(fileTrain.shape[0]):
        imageString.append(folderName + '/' + fileTrain.iloc[i, 0])

    count = 0
    while True:
        if count < len(imageString):
            print('hello')
        else:
            print('hi')
    # 讀檔取圖片名稱及等級
    #imgArray1 = loadImage('label', 'sample_image')
    #imgArray2 = loadImage('dev', 'C1-P1_Dev')
    #imgArray3 = loadImage('train', 'C1-P1_Train')

# VGG 16
#model = VGG16(weights='imagenet', include_top=True)
main()