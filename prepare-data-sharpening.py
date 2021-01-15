import sys
import os
import cv2

def sharpening(path):

    for file in os.listdir(path):

        img = cv2.imread(path + '/' + file)
 
        # sharpening
        blur_img = cv2.GaussianBlur(img, (0, 0), 5)
        sharpened = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)

        # denoising
        denoise = cv2.fastNlMeansDenoisingColored(sharpened, dst=None, h=10, hColor=10, templateWindowSize=3, searchWindowSize=9)

        print("Saving {}".format(file))

        cv2.imwrite("dataset/original-sharpened/{}".format(file), denoise)

sharpening("dataset/original")
sharpening("dataset/test")

