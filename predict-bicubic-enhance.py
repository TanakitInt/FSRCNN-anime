# import specific necessary packages

from keras.models import Sequential, load_model, Input, Model
#from keras.layers import Conv2D, Conv2DTranspose, Input, Activation
#from keras.optimizers import SGD, Adam
import cv2
import numpy as np
import math
import os

# define main prediction function

def predict():
    
    # Display weight summary
    print("Loading trained weight...")
    
    model = load_model('my_model-fsrcnn-anime-tanakitint.h5')
    model.summary()

    # get scale settings from text file ---

    # default bicubic scale is 2

    try:
        settings = open("settings/settings_bicubic.txt", "r")
        bicubic_scale = settings.read()
        BICUBIC_SCALE = float(bicubic_scale)        #stored in variable

    except Exception as e:
        print("Error occured. You might be misconfig the settings, please check the bicubic settings again.")
        print("Please make sure that bicubic setting must not a zero or negative numbers." + '\n')
        print("Error! " + str(e) + '\n')

    INPUT_NAME = "input/1.png"
    OUTPUT_NAME = "output/1-bicubic-enhanced.png"

    # Read image
    img = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)

    # Enlarge image with Bicubic interpolation method
    img = cv2.resize(img, None, fx=BICUBIC_SCALE, fy=BICUBIC_SCALE, interpolation=cv2.INTER_CUBIC)

    # convert from BGR to YCrCb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1], shape[0]), cv2.INTER_CUBIC)
    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.

    # prediction
    pre = model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img[:, : ,0] = pre[0, :, :, 0]

    # convert from YCrCb to BGR and save image
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(OUTPUT_NAME, img)

    # print success!
    print("Prediction success!")

predict()

