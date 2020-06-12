import os
import shutil
import pandas as pds

def splitImage(fileName, folderName):
    file = pds.read_csv(fileName + ".csv")
    if not os.path.isdir(fileName):
        os.mkdir(fileName)
        print(f'Create folder {fileName}')
    if not os.path.isdir(fileName + "/A"):
        os.mkdir(fileName + "/A")
        print(f'Create folder {fileName + "/A"}')
    if not os.path.isdir(fileName + "/B"):
        os.mkdir(fileName + "/B")
        print(f'Create folder {fileName + "/B"}')
    if not os.path.isdir(fileName + "/C"):
        os.mkdir(fileName + "/C")
        print(f'Create folder {fileName + "/C"}')
    for i in range(file.shape[0]):
        originalPath = folderName + "/" + file.iloc[i][0]
        newPath = fileName + "/" + file.iloc[i][1]
        shutil.copy(originalPath, newPath)

fileName = "dev" # Train, Dev
folderName = "C1-P1_Dev" # C1-P1_Train, C1-P1_Dev
splitImage(fileName, folderName)