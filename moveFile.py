import os
import shutil
import pandas as pds

fileName = "dev" # Train, Dev
folderName = "C1-P1_Dev" # C1-P1_Train, C1-P1_Dev
file = pds.read_csv(fileName + ".csv")
os.mkdir(fileName)
os.mkdir(fileName + "/A")
os.mkdir(fileName + "/B")
os.mkdir(fileName + "/C")
for i in range(file.shape[0]):
    originalPath = folderName + "/" + file.iloc[i][0]
    newPath = fileName + "/" + file.iloc[i][1]
    shutil.copy(originalPath, newPath)

#os.mkdir("test")
#shutil.copy()