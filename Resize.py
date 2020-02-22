from os import environ, path, chdir, listdir, mkdir
import sys
from PIL import Image
from pandas import DataFrame, to_datetime
from cv2 import imread, COLOR_RGB2GRAY, ADAPTIVE_THRESH_GAUSSIAN_C, MORPH_OPEN, morphologyEx, getStructuringElement, imwrite, cvtColor, threshold, MORPH_RECT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


chdir(environ['USERPROFILE'] + "\\Desktop\\Whale Challenge\\")

raw_fldr = "raw/"
train_fldr = "train/"
train_resized_fldr = "train_resized/"
train_preprocessed_fldr = "train_preprocessed/"

raw_dirs = listdir(raw_fldr)
df = DataFrame(columns=["Picture", "Date"])

def resize_images():
    for fldr in listdir(train_fldr):
        pics = listdir(train_fldr + str(fldr))
        try:
            mkdir(path.join(train_resized_fldr, fldr))
        except:
            pass
        for pic in pics:
            im = Image.open(path.join(train_fldr, fldr, pic))
            im = im.resize((224,224), Image.ANTIALIAS)
            im.save(path.join(train_resized_fldr, fldr, pic.replace(".jpg", "_resized.jpg")), 'JPEG', quality=90)


def remove_sea():
    for fldr in listdir(train_resized_fldr):
        pics = listdir(train_resized_fldr + str(fldr))
        try:
            mkdir(path.join(train_preprocessed_fldr, fldr))
        except:
            pass
        
        for pic in pics:
            img = imread(path.join(train_resized_fldr, fldr, pic))
            grey = cvtColor(img, COLOR_RGB2GRAY)
            ret, img = threshold(grey,50,255, ADAPTIVE_THRESH_GAUSSIAN_C)
            
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_regular.jpg"), img)




            kernel1 = np.ones((1, 1), np.uint8)
            img1 = morphologyEx(img, MORPH_OPEN, kernel1, iterations=2)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_kernel1.jpg"), img1)

            kernel2 = np.ones((2, 2), np.uint8)
            img2 = morphologyEx(img, MORPH_OPEN, kernel2, iterations=2)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_kernel2.jpg"), img2)

            kernel3 = np.ones((100, 100), np.uint8)
            img3 = morphologyEx(img, MORPH_OPEN, kernel3, iterations=2)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_kernel3.jpg"), img3)



def get_picture_dates():
    for fldr in listdir(raw_fldr):
        for pic in listdir(raw_fldr + fldr):
            if pic.endswith("jpg"):
                # Dates are always in this position but sometimes it's lacking a dash so we take first eight digits of the date as the date
                picdate = pic.split("-")[2][:8]
                df.loc[len(df)] = [pic, picdate]
    df['Date']= to_datetime(df['Date']) 
    print(df)







def main():
    #resize_images()
    remove_sea()






if __name__ == "__main__":
    main()
    
    
    
