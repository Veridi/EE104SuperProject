# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:49:16 2020

@author: Conspiracing
"""

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
"""
was used for testing
myImages = []
"""

###################Part 2 of Project Option 4########################

#Load images from our folder, in target size 32,32 pixels
for i in range(1, 21):
    img = load_img('images/{}.jpg'.format(i), target_size=(32, 32))
    plt.savefig('images/{}.jpg'.format(i + 20))
    #convert the image to an array
    img = img_to_array(img)
    #format the array into the 32x32x3 format
    img = img.reshape(1, 32, 32, 3)
    #format numbers to floats
    img = img.astype('float32')
    #convert the numbers in the array to be from 0 to 1 (not necessary but for array to be smaller values)
    img = img / 255.0
    np.save('{}.file'.format(i), img)
    """
    Code from our trial runs
    #print(img)
    #myImages.append(img)
    #file = open("images/array{}.txt".format(i), "w")
    #file.write(str(img))
    #file.close()
    """
    
#############Part 4 of the Project Option 4 #############################
############# Original Dataset #############
#Here we use the original code in section 1 of the example given
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)
plt.show()
#The code when ran outputs the first five components of the training dataset, as well as their labels.
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

############ Replacing the first 5 images ############
#Here we redo the code, such that we replace teh first five images in the training images, as well as their corresponding label values in the the training labels
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(1, 6):
    #since our values are between 0 and 1, if we input directly into the xtrain array, it recognizes as 0. Thus we must first load into a temp variable
    #and multiply by the 255 factor (RGB) to get it back to a 0 to 255 scale
    testX = np.load('{}.file.npy'.format(i))
    testX = testX * 255
    X_train[i - 1] = testX     
    img = X_train[i - 1]
    #we replot the dataset using our images inputted into the dataset
    axarr[i - 1].imshow(img)
plt.show()

############ Replacing the first 5 labels ############
y_train[0] = 0
y_train[1] = 0
y_train[2] = 1
y_train[3] = 1
y_train[4] = 2

print('Replaced testing images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))
