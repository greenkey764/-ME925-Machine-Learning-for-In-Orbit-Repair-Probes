import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os
import random
import datetime

num_classes = 3

image_size = 240
image_flat = image_size * image_size
image_shape = (image_size, image_size)
image_channels = 1

learn_rate = 0.0005
epochs = 4
batches = 9
sub_batches = 20

input = tf.placeholder(tf.float32, [None, image_size, image_size])
output = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

def convLayer(x, w, b, s = 1):

    x = tf.nn.conv2d(x, w, strides = [1, s, s, 1], padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)

    return x

def maxPool(x, k = 2):

    x = tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = 'SAME')

    return x

weights = {
    'w1' : tf.Variable(tf.random_normal([5, 5, 1, 32]), name = 'w1'),
    'w2' : tf.Variable(tf.random_normal([5, 5, 32, 64]), name = 'w2'),
    'w3' : tf.Variable(tf.random_normal([3, 3, 64, 128]), name = 'w3'),
    'w4' : tf.Variable(tf.random_normal([3, 3, 128, 256]), name = 'w4'),
    'wf' : tf.Variable(tf.random_normal([15 * 15 * 256, 4096]), name = 'wf'),
    'wo' : tf.Variable(tf.random_normal([4096, num_classes]), name = 'wo')
}

biases = {
    'b1' : tf.Variable(tf.random_normal([32]), name = 'b1'),
    'b2' : tf.Variable(tf.random_normal([64]), name = 'b2'),
    'b3' : tf.Variable(tf.random_normal([128]), name = 'b3'),
    'b4' : tf.Variable(tf.random_normal([256]), name = 'b4'),
    'bf' : tf.Variable(tf.random_normal([4096]), name = 'bf'),
    'bo' : tf.Variable(tf.random_normal([num_classes]), name = 'bo')
}

def Network(x, w, b, prob):

    x = tf.reshape(x, shape = [-1, 240, 240, 1])

    conv1 = convLayer(x, w['w1'], b['b1'], 1)
    conv1 = maxPool(conv1, 2)

    conv2 = convLayer(conv1, w['w2'], b['b2'], 1)
    conv2 = maxPool(conv2, 2)

    conv3 = convLayer(conv2, w['w3'], b['b3'])
    conv3 = maxPool(conv3, 2)

    conv4 = convLayer(conv3, w['w4'], b['b4'])
    conv4 = maxPool(conv4, 2)

    flat1 = tf.reshape(conv4, [-1, w['wf'].get_shape().as_list()[0]])
    flat1 = tf.add(tf.matmul(flat1, w['wf']), b['bf'])
    flat1 = tf.nn.relu(flat1)

    flat1 = tf.nn.dropout(flat1, prob)

    classification = tf.add(tf.matmul(flat1, w['wo']), b['bo'])

    return classification

model = Network(input, weights, biases, keep_prob)

costFnc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = output, logits = model))
opFnc = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(costFnc)

actual_class = tf.argmax(output, 1)
predicted_class = tf.argmax(model, 1)

check = tf.equal(actual_class, predicted_class)

accuracyFnc = tf.reduce_mean(tf.cast(check, tf.float32))

initialize_all = tf.initializers.global_variables()

trained_loss = []
trained_acc = []

valid_acc = []

tracked_steps = []

outer_test_x = []
outer_test_y = []

callSave = tf.train.Saver()

with tf.Session() as sess:

    step = 0

    broken = False

    sess.run(initialize_all)

    print('\n----- Training Initiated -----')

    for epoch in range(epochs):
        for outer_batch in range(batches):

            current_val_x = []
            current_val_y = []

            current_train_x = np.load('../Training Data/Data/batch_' + str(outer_batch + 1) + '/train_images.npy')
            current_train_y = np.load('../Training Data/Data/batch_' + str(outer_batch + 1) + '/train_labels.npy')

            for inner_batch in range(sub_batches):

                print('Sub-Batch: {} / {}'.format(inner_batch + 1, sub_batches))
                print('-'*30)

                sess.run(opFnc, feed_dict = {input : current_train_x[int((inner_batch)*(1440/20)) : int((inner_batch + 1)*(1440/20))],\
                output : current_train_y[int((inner_batch)*(1440/20)) : int((inner_batch + 1)*(1440/20))], keep_prob : 0.5})

            tr_acc = sess.run(accuracyFnc, feed_dict = {input : current_train_x[314 : 414], output : current_train_y[314 : 414], keep_prob : 1})

            current_train_x = []
            current_train_y = []

            current_val_x = np.load('../Training Data/Data/batch_' + str(outer_batch + 1) + '/valid_images.npy')
            current_val_y = np.load('../Training Data/Data/batch_' + str(outer_batch + 1) + '/valid_labels.npy')

            vl_acc = sess.run(accuracyFnc, feed_dict = {input : current_val_x[:100], output : current_val_y[:100], keep_prob : 1})

            print('Epoch: {} / {} (Batch: {} / {})'.format(epoch + 1, epochs, outer_batch + 1, batches))
            print('Training Accuracy: {}'.format(tr_acc))
            print('Validation Accuracy: {}'.format(vl_acc))
            print('-'*30)

            step += batches

            trained_acc.append(tr_acc)
            valid_acc.append(vl_acc)
            tracked_steps.append(step)

            if len(trained_acc) > 4:
                if all(r > 0.86 for r in valid_acc[-3:-1]) and all(o > 0.86 for o in trained_acc[-3:-1]):

                    broken = True
                    break

        if broken == True:
            break

    current_val_x = []
    current_val_y = []

    outer_test_x = np.load('../Training Data/Data/batch_10/valid_images.npy')
    outer_test_y = np.load('../Training Data/Data/batch_10/valid_labels.npy')

    class_prediction = sess.run(predicted_class, feed_dict = {input : outer_test_x[:100], output: outer_test_y[:100], keep_prob : 1})

    callSave.save(sess, 'C:/Users/Andrew/Documents/University/5th Year/Autonomous Sensing, Learning, and Reasoning/Project/Results/Second Model/second_model')

    np.save('C:/Users/Andrew/Documents/University/5th Year/Autonomous Sensing, Learning, and Reasoning/Project/Results/Second Model/trained_acc.npy', trained_acc)
    np.save('C:/Users/Andrew/Documents/University/5th Year/Autonomous Sensing, Learning, and Reasoning/Project/Results/Second Model/valid_acc.npy', valid_acc)
    np.save('C:/Users/Andrew/Documents/University/5th Year/Autonomous Sensing, Learning, and Reasoning/Project/Results/Second Model/tracked_steps.npy', tracked_steps)

tracked_steps = np.insert(tracked_steps, 0, 0)
trained_acc = np.insert(trained_acc, 0, 0)
valid_acc = np.insert(valid_acc, 0, 0)

trained_acc = trained_acc * 100
valid_acc = valid_acc * 100

test_check = np.argmax(outer_test_y[:100], axis = 1)

hit_rate = (test_check == class_prediction)

hit_rate = hit_rate.sum()

print('Test Accuracy: ' + str(hit_rate) + '%')

for pred in range(50):

    if test_check[pred] == 0:
        this_label = 'Antenna'

    elif test_check[pred] == 1:
        this_label = 'Communications Dish'

    elif test_check[pred] == 2:
        this_label = 'Solar Array'

    else:
        this_label = 'ERROR'

    if class_prediction[pred] == 0:
        this_logit = 'Antenna'

    elif class_prediction[pred] == 1:
        this_logit = 'Communications Dish'

    elif class_prediction[pred] == 2:
        this_logit = 'Solar Array'

    else:
        this_logit = 'ERROR'

    fig = plt.figure()
    fig.suptitle('Prediction: ' + this_logit + '   |   Label: ' + this_label)
    plt.imshow(np.array((outer_test_x[pred]*255), dtype = 'uint8'), cmap = 'gray')
    plt.show()


plt.xticks(tracked_steps)
plt.yticks(np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]) * 100)
plt.autoscale(False)

for xc in tracked_steps[::9]:
    if xc == 0:
        plt.axvline(x=xc, color = (0.4, 0.4, 0.4), linestyle='--', label = 'Batch')
    else:
        plt.axvline(x=xc, color = (0.4, 0.4, 0.4), linestyle='--')

plt.plot(tracked_steps, len(tracked_steps)*[95], color = (0.4, 0.7, 0.4), linestyle = '-.', label = 'Threshold')

plt.plot(tracked_steps, trained_acc, 'b-', label = 'Training Accuracy')
plt.plot(tracked_steps, valid_acc, 'r-', label = 'Cross-Validation Accuracy')

plt.xlabel('Batch (1 Epoch = 180 Mini-batches)')
plt.ylabel('Accuracy (%)')
plt.legend(loc = 10, frameon = True, fontsize = 11, ncol = 2, framealpha = 1, facecolor = (1, 1, 1))

plt.show()
