import numpy as np

# array = np.array([[1, 2], [6, 4], [2, 1], [3, 4], [7, 1], [2, 9]])
#
# print(array)
#
# print('\n')
#
# np.random.shuffle(array)
#
# print(array)


def shuffling():

    for num1 in range(10):

        train_images = np.load('../Training Data/Data/batch_' + str(num1 + 1) + '/training_images.npy')
        train_labels = np.load('../Training Data/Data/batch_' + str(num1 + 1) + '/training_labels.npy')
        train_array = list(zip(train_images, train_labels))

        val_images = np.load('../Training Data/Data/batch_' + str(num1 + 1) + '/validation_images.npy')
        val_labels = np.load('../Training Data/Data/batch_' + str(num1 + 1) + '/validation_labels.npy')
        val_array = list(zip(val_images, val_labels))

        np.random.shuffle(train_array)

        np.random.shuffle(val_array)

        train_images, train_labels = zip(*train_array)

        np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/train_images', train_images)
        np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/train_labels', train_labels)

        val_images, val_labels = zip(*val_array)

        print('-'*20)
        print('Batch ' + str(num1) + ' Shuffled!')
        print('-'*20)

        np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/valid_images', val_images)
        np.save('../Training Data/Data/batch_' + str(num1 + 1) + '/valid_labels', val_labels)

shuffling()

example_x = np.load('../Training Data/Data/batch_6/train_images.npy')
example_y = np.load('../Training Data/Data/batch_6/train_labels.npy')

print(np.shape(example_x))
print(np.shape(example_y))
