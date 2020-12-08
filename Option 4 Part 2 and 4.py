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

myImages = []

for i in range(1, 21):
    img = load_img('images/{}.jpg'.format(i), target_size=(32, 32))
    plt.savefig('images/{}.jpg'.format(i + 20))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    #print(img)
    np.save('{}.file'.format(i), img)
    #myImages.append(img)
    #file = open("images/array{}.txt".format(i), "w")
    #file.write(str(img))
    #file.close()
    
############# Original Dataset #############

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(5):
    img = X_train[i]
    axarr[i].imshow(img)
plt.show()

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print('Example training images and their labels: ' + str([x[0] for x in y_train[0:5]]))
print('Corresponding classes for the labels: ' + str([cifar_classes[x[0]] for x in y_train[0:5]]))

############ Replacing the first 5 images ############

f, axarr = plt.subplots(1, 5)
f.set_size_inches(16, 6)
for i in range(1, 6):
    testX = np.load('{}.file.npy'.format(i))
    testX = testX * 255
    X_train[i - 1] = testX     #Multiply by 255 because the values returned from np.load were a percentage of the RGB value
    img = X_train[i - 1]
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