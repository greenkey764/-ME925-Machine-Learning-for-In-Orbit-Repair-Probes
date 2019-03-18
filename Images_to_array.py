import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from random import randint

training_path = "../Training Data/Images/training/"

validation_path = "../Training Data/Images/validation/"

def one_hot_labelling(image):

    label = image.split('_')[0]

    if label == 'antenna':
        hot = np.array([1, 0, 0])

    if label == 'dish':
        hot = np.array([0, 1, 0])

    if label == 'solar':
        hot = np.array([0, 0, 1])

    return hot

def data_generation(dir, start, stop):

    images = []
    labels = []

    for i in range(len(os.listdir(dir))):
        path = dir + os.listdir(dir)[i] + '/'

        for j in range(int(start), int(stop)): #Make file batches of total number / 10 images
            image_name = path + os.listdir(path)[j]
            image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
            image = np.array(image)/255
            images.append(image)
            labels.append(one_hot_labelling(os.listdir(path)[j]))
    return images, labels

for num1 in range(10):

    training_images, training_labels = data_generation(training_path, num1*(14400/30), (num1 + 1)*(14400/30))
    validation_images, validation_labels = data_generation(validation_path, num1*(3600/30), (num1 + 1)*(3600/30))

    print('-'*20)
    print('Batch ' + str(num1 + 1) + ' Complete!')
    print('-'*20)

    np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/training_images', training_images)
    np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/training_labels', training_labels)
    np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/validation_data', validation_images)
    np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/validation_labels', validation_labels)

plotted = randint(0,1440)

fig = plt.figure()
fig.suptitle(str(training_images[plotted]))
plt.imshow(np.array((training_images[plotted]*255), dtype = 'uint8'), cmap = 'gray')
plt.show()
