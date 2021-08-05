import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

import random
import cv2

import imutils




categories = ["cracked", "uncracked"]

from tensorflow.keras.models import load_model
model = load_model('D:/Crack_detection/crack_model_1.h5')

path = input("enter the image path:  ")

IMG_SIZE = 100


def prepare(filepath):
    #IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath)  # read in the image
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    print(img_array)
    img_array = np.array(img_array)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # ret
    new_array = new_array/255.0
    return new_array



    
    
print("your prediction is:   ")

prediction = model.predict(prepare(path))
print(prediction[0][0])

cracked_prob = prediction[0][0]

uncracked_prob = prediction[0][1]

print(prediction[0])

result = np.round(prediction[0]) 


for i in range(0, len(result)):
    if(i==0 and result[i]==1.0):
       new_path = path
       img = cv2.imread(new_path)
       plt.imshow(img)
       plt.show()
       print("The Model predicts that the image is Cracked with probability {}%".format(cracked_prob * 100))
       im = cv2.imread(new_path)
       imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
       
       # create a binary thresholded image
      # _, binary = cv2.threshold(imgray, 225, 255, cv2.THRESH_BINARY_INV)
       # show it
       #plt.imshow(binary, cmap="gray")
      #plt.show()
       
       
       ret, thresh = cv2.threshold(imgray, 75, 255, 0)
       contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
       
       cv2.drawContours(im, contours, -1, (0,255,0), 2)
       
      

       plt.imshow(im)
       plt.show()
    if(i==1 and result[i]==1.0):
        new_path = path
        img = cv2.imread(new_path)
        plt.imshow(img)
        plt.show()
        print("The model predicts that the image is uncracked with probability {}%".format(uncracked_prob * 100))
        

    