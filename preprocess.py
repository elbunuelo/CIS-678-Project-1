'''
GVSU CIS-678 Project #1

Script to perform preprocessing of the input data.

This script achieves two tasks: It normalizes the input data so that it falls
in the range -1, 1 and joins the test data and labels so that the test file
has the same structure as the training one.
'''
import numpy as np

'''All relevant files are read and we construct four different arrays,
two of them contain the actual data and the other two contain the corresponding
labels'''
test_labels = np.genfromtxt('data/test_labels.csv', skip_header=1, dtype=np.int8)
test_labels.shape = (len(test_labels), 1)
train_data = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1, dtype=np.int8)
test_pixels = np.genfromtxt('data/test_data.csv', delimiter=',', skip_header=1, dtype=np.int8)
train_pixels = train_data[:, 1:]
train_labels = train_data[:, 0].astype(np.int8)
train_labels.shape = (len(train_labels), 1)

'''
Data normalization must be done on all of the elements in the dataset
(training + testing). For each eleement, We subtract the mean and divide it by
the variance.
'''
all_pixels = np.concatenate((train_pixels, test_pixels))
pixels_mean = np.mean(all_pixels)
pixels_var = np.var(all_pixels)

normalized_pixels = (all_pixels - pixels_mean)/pixels_var

'''After normalizing, we add back the corresponding labels to each of the data
arrays and output them to csv files'''
normalized_training = np.concatenate(
        (train_labels, normalized_pixels[:len(train_pixels),:]), axis=1)
normalized_testing = np.concatenate(
        (test_labels, normalized_pixels[len(train_pixels):, :]), axis=1)

np.savetxt(
        'data/normalized_training_reduced.csv', np.around(normalized_training, decimals=5), delimiter=',', fmt='%1.5f')
np.savetxt(
        'data/normalized_testing_reduced.csv', np.around(normalized_testing, decimals=5), delimiter=',', fmt='%1.5f')
