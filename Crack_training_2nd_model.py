import pickle
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import pandas as pd
import io  
from tqdm import tqdm
#import sklearn.model_selection as sk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from sklearn import model_selection
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
#import tensorflow.keras.utils.to_categorical as to_cat
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

CATEGORIES = ["Major_Crack", "Minor_Crack"]

DATADIR = "D:/crack_detection/train/Major_minor_dataset/"

training_data = []
output_empty = [0] * len(CATEGORIES)

IMG_SIZE = 100

def create_training_data():
    
    for category in CATEGORIES: 
        

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  

        for img in tqdm(os.listdir(path)):  
            
                
            img_array = cv2.imread(os.path.join(path,img))  # convert to array
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
               
            new_array = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            #new_array = np.array(new_array)
                
            output_row = list(output_empty)
    
            output_row[(class_num)] = 1
            #adding the output row to the training set
            training_data.append([new_array, output_row])  # add this to our training_data
            #new_counter = new_counter + 1
                
                
            plt.imshow(img_rgb)
            plt.show
            print("This is class_num for above image {}".format(class_num))
            print("This is the shape for above image with class number:   {},  {}".format(class_num, new_array.shape))



#creatinng the training data
create_training_data()

print(training_data[1])

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[0])
    print(sample[1])
    
X = []
y = []

#train_x = list(training_data[:,0])
#print(train_x)
#train_y = list(training_data[:,1])

for features,label in training_data:
    X.append(features)
    y.append(label)
    
    

X = np.array(X)
y = np.array(y)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

X = X/255.0

print(X.shape)
print(y.shape)

model = Sequential()

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',
                 input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-2
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-3
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-4
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
#Block-5
model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#Block-6
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#Block-7
model.add(Dense(2,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy',
              #optimizer= Adam(lr=0.25, decay=1e-6),
              #metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])





history = model.fit(X, y, batch_size=35, epochs=15, validation_split=0.2, verbose=1, shuffle=False)

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,15)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(0,15)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



model.save('crack_model_2nd.h5', history)
print("model saved")

