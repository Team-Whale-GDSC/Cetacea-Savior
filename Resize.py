from os import environ, path, chdir, listdir, mkdir
import sys
from PIL import Image
from pandas import DataFrame, to_datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


chdir(environ['USERPROFILE'] + "\\Desktop\\Whale Challenge\\")

raw_fldr = "raw/"
train_fldr = "train/"
train_resized_fldr = "train_resized/"
train_preprocessed_fldr = "train_preprocessed/"
raw_dirs = listdir(raw_fldr)


img_shape    = (224,224, 1) # The image shape used by the model


def resize_images():
    for fldr in listdir(train_fldr):
        pics = listdir(train_fldr + str(fldr))
        try:
            mkdir(path.join(train_resized_fldr, fldr))
        except:
            pass
        for pic in pics:
            im = Image.open(path.join(train_fldr, fldr, pic))
            im = im.resize((256,256), Image.ANTIALIAS)
            im.save(path.join(train_resized_fldr, fldr, pic), 'JPEG', quality=95)



def resize_images_v2():
    trig = 1
    for fldr in listdir(train_fldr):
        pics = listdir(train_fldr + str(fldr))
        try:
            mkdir(path.join(train_resized_fldr, fldr))
        except:
            pass
        for pic in pics:
            if trig == 1:
            
                
                img = cv2.imread(path.join(train_fldr, fldr, pic),0)
                canny_edges = cv2.Canny(img,1,10)
                plt.imshow(canny_edges)
                trig = 0
                plt.show()
                '''
                im = im.resize((224,224), Image.ANTIALIAS)
                
                im = crop(path.join(train_fldr, fldr, pic))
                im.save(path.join(train_resized_fldr, fldr, pic), 'JPEG')
                '''



def remove_sea():
    for fldr in listdir(train_resized_fldr):
        pics = listdir(train_resized_fldr + str(fldr))
        try:
            mkdir(path.join(train_preprocessed_fldr, fldr))
        except:
            pass
        
        for pic in pics:
            img = cv2.imread(path.join(train_resized_fldr, fldr, pic))
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, img = cv2.threshold(grey, 50, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            ret, img = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            #cv2.imwrite(path.join(train_preprocessed_fldr, fldr, pic), img)

            
            #imwrite(path.join(train_preprocessed_fldr, fldr, pic + "_regular.jpg"), img)


            #acceptable kernels: (kernel1 = np.ones((1, 1), np.uint8))
            
            kernel1 = np.empty([1, 1]) 
            kernel1 = kernel1.fill(1)
            img1 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1, iterations=1)
            cv2.imwrite(path.join(train_preprocessed_fldr, fldr, pic), img1)

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

from numpy import asarray

def get_train_df():
    pic_df_tmp = DataFrame(columns = ["pic_name", "pic_array"])
    train_df = DataFrame(columns=["pic_name", "id", "date", "d_or_a"])
    for fldr in listdir(raw_fldr):
        for pic in listdir(raw_fldr + fldr):
            if pic.endswith(".jpg"):
                picdate = pic.split("-")[2][:8]
                train_df.loc[len(train_df)] = [pic, "", picdate, 1]
    train_df['date'] = to_datetime(train_df['date'])
    
    for fldr in listdir(train_preprocessed_fldr):
        whale_id = str(fldr)
        for pic in listdir(train_preprocessed_fldr + fldr):
            train_df.loc[train_df.pic_name == pic, 'id'] = whale_id
            image = Image.open(path.join(train_preprocessed_fldr, fldr, pic))
            pic_df_tmp.loc[len(pic_df_tmp)] = [pic, asarray(image)]

    train_df = train_df.merge(pic_df_tmp, how= "outer", on="pic_name")
    print(train_df.head(20))

    train_df.to_pickle("pckle")


def main():
    resize_images()
    remove_sea()
    get_train_df()
    




















if __name__ == "__main__":
    main()
    
    
    
