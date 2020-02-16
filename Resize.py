from os import environ, path, chdir, listdir, mkdir
import sys
from PIL import Image
from pandas import DataFrame, to_datetime
import cv2
import numpy as np


chdir(environ['USERPROFILE'] + "\\Desktop\\Whale Challenge\\")

raw_fldr = "raw/"
raw_dirs = listdir(raw_fldr)
train_fldr = "train/"
raw_train_dirs = listdir(raw_fldr)
train_resized_fldr = "train_resized"
df = DataFrame(columns=["Picture", "Date"])

def modify_images():
    for folder in raw_train_dirs:
        pics = listdir(train_fldr + str(folder))
        try:
            mkdir(path.join(train_resized_fldr, folder))
        except:
            pass
        for pic in pics:
            im = Image.open(path.join(train_fldr, folder, pic))
            im = im.resize((224,224), Image.ANTIALIAS)
            im = im.convert('1')
            im.save(path.join(train_resized_fldr, folder, pic.replace(".jpg", "_resized.jpg")), 'JPEG', quality=90)
        

def get_picture_dates():
    for folder in raw_train_dirs:
        for pic in listdir(raw_fldr + folder):
            if pic.endswith("jpg"):
                # Dates are always in this position but sometimes it's lacking a dash so we take first eight digits of the date as the date
                picdate = pic.split("-")[2][:8]
                df.loc[len(df)] = [pic, picdate]
    df['Date']= to_datetime(df['Date']) 
    print(df)



def remove_sea():
    img = cv2.imread("train_resized/-1/PM-WWA-20050513-243_resized.jpg")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (100, 0, 0), (360, 100,50))

    ## slice the green
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    ## save 
    cv2.imwrite("green_test.png", green)

if __name__ == "__main__":
    remove_sea()
