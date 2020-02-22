from os import environ, path, chdir, listdir, mkdir
import sys
from PIL import Image
from pandas import DataFrame, to_datetime
from cv2 import imread, COLOR_RGB2GRAY, ADAPTIVE_THRESH_GAUSSIAN_C, MORPH_OPEN, morphologyEx, getStructuringElement, imwrite, cvtColor, threshold, MORPH_RECT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

chdir(environ['USERPROFILE'] + "\\Desktop\\Whale Challenge\\")

raw_fldr = "raw/"
train_fldr = "train/"
train_resized_fldr = "train_resized/"
train_preprocessed_fldr = "train_preprocessed/"

raw_dirs = listdir(raw_fldr)

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
            im.save(path.join(train_resized_fldr, fldr, pic), 'JPEG', quality=90)


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
            
            #imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_regular.jpg"), img)


            #acceptable kernels: (kernel1 = np.ones((1, 1), np.uint8))

            kernel1 = np.empty([1, 1]) 
            kernel1 = kernel1.fill(1)
            img1 = morphologyEx(img, MORPH_OPEN, kernel1, iterations=1)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic), img1)

            '''
            kernel2 = np.empty([1, 1]) 
            kernel2 = kernel2.fill(2)
            img2 = morphologyEx(img, MORPH_OPEN, kernel2, iterations=1)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_kernel2.jpg"), img2)
            
            
            kernel3 = np.empty([1, 1]) 
            kernel3 = kernel3.fill(100)

            img3 = morphologyEx(img, MORPH_OPEN, kernel3, iterations=1)
            imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_kernel3.jpg"), img3)
            '''


def get_picture_date_id_state():
    train_df = DataFrame(columns=["picture", "id", "date", "d_or_a"])
    for fldr in listdir(raw_fldr):
        for pic in listdir(raw_fldr + fldr):
            if pic.endswith(".jpg"):
                picdate = pic.split("-")[2][:8]
                train_df.loc[len(train_df)] = [pic, picdate, 1]
    train_df['date'] = to_datetime(train_df['date'])
    
    
    for fldr in listdir(train_preprocessed_fldr):
        whale_id = str(fldr)
        for pic in listdir(train_preprocessed_fldr + fldr):
            train_df[train_df.picture == pic] = whale_id

    print(train_df.head(2))




'''
def model(train_df):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_dataframe(train_df, 
                    x_col='picture', 
                    y_col='animal', 
                    target_size=(100,100), #images resized
                    batch_size=20,
                    class_mode='binary')


'''















def main():
    #resize_images()
    #remove_sea()
    get_picture_dates_and_location()
    #model(train_df)





















if __name__ == "__main__":
    main()
    
    
    
