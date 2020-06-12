import os
import shutil
import pandas as pds

def splitImage(fileName, folderName):
    file = pds.read_csv(fileName + ".csv")
    if not os.path.isdir(fileName):
        os.mkdir(fileName)
        print(f'Create folder {fileName}')
    for i in range(file.shape[0]):
        if not os.path.isdir(fileName + "/" + file.iloc[i][1]):
            os.mkdir(fileName + "/" + file.iloc[i][1])
            print(f'Create folder {fileName + "/" + file.iloc[i][1]}')
        # 取得檔案
        originalPath = folderName + "/" + file.iloc[i][0]
        # 分到該 Label 目錄下
        newPath = fileName + "/" + file.iloc[i][1]
        shutil.copy(originalPath, newPath)

fileName = "dev" # Train, Dev
folderName = "C1-P1_Dev" # C1-P1_Train, C1-P1_Dev
splitImage(fileName, folderName)