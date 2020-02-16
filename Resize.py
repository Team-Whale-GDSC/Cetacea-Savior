from os import environ, path, chdir, listdir
import sys
from PIL import Image

chdir(environ['USERPROFILE'] + "\\Desktop\\Whale Challenge\\")


train_fldr = "train/"
train_resized_fldr = "train_resized"
raw_train_dirs = listdir(train_fldr)



def modify_image():
    for folder in raw_train_dirs:
        pics = listdir(train_fldr + str(folder))
        for pic in pics:
            im = Image.open(path + folder + "\\" + pic)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(path.join(train_resized_fldr, folder, pic.replace(".jpg", "_resized.jpg"), 'JPEG', quality=90)

modify_image()