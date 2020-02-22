from os import environ, path, chdir, listdir, mkdir
import sys
from PIL import Image
from pandas import DataFrame, to_datetime
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import preprocessing
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras import backend as K

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
            im = im.resize((224,224), Image.ANTIALIAS)
            im.save(path.join(train_resized_fldr, fldr, pic), 'JPEG', quality=90)

def crop(PATH):
    h_factor = 2 #here is for 1:"2"
    img = cv2.imread(PATH,0)
    
    v = img.shape[0] # vertial pixels
    h = img.shape[1] # horizontal pixels 

    #find edges with Canny algorism
    canny_edges = cv2.Canny(img,300,300)
    if v < h/h_factor:
        fill_length = int(abs(h/h_factor-v)*0.5)#np.random.rand()) # for upper filling
    
        fill = np.zeros(fill_length* h).reshape(fill_length, h) # black rectangle for upper filling

        canny_edges = np.r_[fill,canny_edges,fill] # fill with black rectangle
        img = np.r_[fill,img,fill] # fill with black rectangle
    
    ver = int(h/h_factor)
    cnt = []
    for i in range(canny_edges.shape[0]-ver+2):
        cnt.append(canny_edges[i:i+ver,:].sum()/255)

    cnt_arr = np.array(cnt)
    i = cnt_arr.argmax()
    return Image.fromarray(np.uint8(img[i:i+ver,:]))



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


def get_picture_date_id_state():
    train_df = DataFrame(columns=["picture", "id", "date", "d_or_a"])
    for fldr in listdir(raw_fldr):
        for pic in listdir(raw_fldr + fldr):
            if pic.endswith(".jpg"):
                picdate = pic.split("-")[2][:8]
                train_df.loc[len(train_df)] = [pic, "d", picdate, 1]
    train_df['date'] = to_datetime(train_df['date'])
    
    
    for fldr in listdir(train_preprocessed_fldr):
        whale_id = str(fldr)
        for pic in listdir(train_preprocessed_fldr + fldr):
            train_df.loc[train_df.picture == pic, 'id'] = whale_id

    print(train_df.head(20))





def model(train_df):
    train_datagen = preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_dataframe(train_df, 
                    x_col='picture', 
                    y_col='id', 
                    batch_size=20,
                    class_mode='binary')

def subblock(x, filter, **kwargs):
    x = BatchNormalization()(x)
    y = x
    y = Conv2D(filter, (1, 1), activation='relu', **kwargs)(y) # Reduce the number of features to 'filter'
    y = BatchNormalization()(y)
    y = Conv2D(filter, (3, 3), activation='relu', **kwargs)(y) # Extend the feature field
    y = BatchNormalization()(y)
    y = Conv2D(K.int_shape(x)[-1], (1, 1), **kwargs)(y) # no activation # Restore the number of original features
    y = Add()([x,y]) # Add the bypass connection
    y = Activation('relu')(y)
    return y

def build_model(lr, l2, activation='sigmoid'):

    ##############
    # BRANCH MODEL
    ##############
    regul  = regularizers.l2(l2)
    optim  = Adam(lr=lr)
    kwargs = {'padding':'same', 'kernel_regularizer':regul}

    inp = Input(shape=img_shape) # 384x384x1
    x   = Conv2D(64, (9,9), strides=2, activation='relu', **kwargs)(inp)

    x   = MaxPooling2D((2, 2), strides=(2, 2))(x) # 96x96x64
    for _ in range(2):
        x = BatchNormalization()(x)
        x = Conv2D(64, (3,3), activation='relu', **kwargs)(x)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 48x48x64
    x = BatchNormalization()(x)
    x = Conv2D(128, (1,1), activation='relu', **kwargs)(x) # 48x48x128
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 24x24x128
    x = BatchNormalization()(x)
    x = Conv2D(256, (1,1), activation='relu', **kwargs)(x) # 24x24x256
    for _ in range(4): x = subblock(x, 64, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 12x12x256
    x = BatchNormalization()(x)
    x = Conv2D(384, (1,1), activation='relu', **kwargs)(x) # 12x12x384
    for _ in range(4): x = subblock(x, 96, **kwargs)

    x = MaxPooling2D((2, 2), strides=(2, 2))(x) # 6x6x384
    x = BatchNormalization()(x)
    x = Conv2D(512, (1,1), activation='relu', **kwargs)(x) # 6x6x512
    for _ in range(4): x = subblock(x, 128, **kwargs)
    
    x             = GlobalMaxPooling2D()(x) # 512
    branch_model  = Model(inp, x)
    
    ############
    # HEAD MODEL
    ############
    mid        = 32
    xa_inp     = Input(shape=branch_model.output_shape[1:])
    xb_inp     = Input(shape=branch_model.output_shape[1:])
    x1         = Lambda(lambda x : x[0]*x[1])([xa_inp, xb_inp])
    x2         = Lambda(lambda x : x[0] + x[1])([xa_inp, xb_inp])
    x3         = Lambda(lambda x : K.abs(x[0] - x[1]))([xa_inp, xb_inp])
    x4         = Lambda(lambda x : K.square(x))(x3)
    x          = Concatenate()([x1, x2, x3, x4])
    x          = Reshape((4, branch_model.output_shape[1], 1), name='reshape1')(x)

    # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
    x          = Conv2D(mid, (4, 1), activation='relu', padding='valid')(x)
    x          = Reshape((branch_model.output_shape[1], mid, 1))(x)
    x          = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
    x          = Flatten(name='flatten')(x)
    
    # Weighted sum implemented as a Dense layer.
    x          = Dense(1, use_bias=True, activation=activation, name='weighted-average')(x)
    head_model = Model([xa_inp, xb_inp], x, name='head')

    ########################
    # SIAMESE NEURAL NETWORK
    ########################
    # Complete model is constructed by calling the branch model on each input image,
    # and then the head model on the resulting 512-vectors.
    img_a      = Input(shape=img_shape)
    img_b      = Input(shape=img_shape)
    xa         = branch_model(img_a)
    xb         = branch_model(img_b)
    x          = head_model([xa, xb])
    model      = Model([img_a, img_b], x)
    model.compile(optim, loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])
    return model, branch_model, head_model






def main():
    #resize_images()
    #resize_images_v2()

    #remove_sea()
    #get_picture_date_id_state()
    #model(train_df)
    model, branch_model, head_model = build_model(64e-5,0)
    head_model.summary()





















if __name__ == "__main__":
    main()
    
    
    
