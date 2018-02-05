#! /usr/local/bin/python3
import math
import numpy as np

test_labels = np.genfromtxt('test_labels.csv', skip_header=1)
test_labels.shape = (len(test_labels), 1)
train_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)
test_pixels = np.genfromtxt('test_data.csv', delimiter=',', skip_header=1)
train_pixels = train_data[:, 1:]
train_labels = train_data[:, 0].astype(np.int8)
train_labels.shape = (len(train_labels), 1)

all_pixels = np.concatenate((train_pixels, test_pixels))
pixels_mean = np.mean(all_pixels)
pixels_var = np.var(all_pixels)

normalized_pixels = (all_pixels - pixels_mean)/pixels_var


train_pixels_rows = train_pixels.shape[0]
train_pixels_columns = train_pixels.shape[1]


normalized_rows = train_pixels_rows
normalized_columns = 1 + train_pixels_columns

normalized_training = np.concatenate((train_labels, normalized_pixels[:len(train_pixels),:]), axis=1)
normalized_testing = np.concatenate((test_labels, normalized_pixels[len(train_pixels):, :]), axis=1)

np.savetxt('normalized_training.csv', normalized_training, delimiter=',')
np.savetxt('normalized_testing.csv', normalized_testing, delimiter=',')
