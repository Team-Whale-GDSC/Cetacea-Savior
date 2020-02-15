from os import environ, path, chdir, listdir
import sys
from PIL import Image

chdir(environ['USERPROFILE'] + "\Desktop\\Whale Challenge\\")


path = "train/"
dirs = listdir(path)

def modify_image():
    for folder in dirs:
        pics = listdir(path + str(folder))
        for pic in pics:
            print(folder + "-" + pic)
            #im = Image.open(path + folder + pic)


modify_image()