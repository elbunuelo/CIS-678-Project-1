import argparse
from neural_network import NeuralNetwork
import pickle
import os
from time import time
import numpy as np

HIDDEN_NODES = 20
OUTPUT_NUMBER = 10
ETA = 0.3
ITERATIONS = 1000
VALIDATION_DATA_PROPORTION = 0.2
VALIDATION_THRESHOLD = 0.001

parser = argparse.ArgumentParser(description='Test the neural network')
parser.add_argument('--training_file', type=str, required=True,
        help='File containing the training cases for the neural network')
parser.add_argument('--testing_file', type=str, required=True,
        help='File containing the testing cases for the neural network')
args = parser.parse_args()

train_pickle = 'pickle/{}.pickle'.format(args.training_file.split('/').pop())
test_pickle = 'pickle/{}.pickle'.format(args.testing_file.split('/').pop())

# Read all data files
read_start_time = time()
if os.path.isfile(train_pickle):
    train_pickle_file = open(train_pickle, 'rb')
    train_data = pickle.load(train_pickle_file)
else:
    train_pickle_file = open(train_pickle, 'wb')
    train_data = np.genfromtxt(args.training_file, delimiter=',')
    pickle.dump(train_data, train_pickle_file)


if os.path.isfile(test_pickle):
    test_pickle_file = open(test_pickle, 'rb')
    test_data = pickle.load(test_pickle_file)
else:
    test_pickle_file = open(test_pickle, 'wb')
    test_data = np.genfromtxt(args.testing_file, delimiter=',')
    pickle.dump(test_data, test_pickle_file)

run_time_results = np.zeros(7)
run_success_results = np.zeros(7)
for i, iterations in enumerate([5, 10, 20, 40, 80, 100, 1000]):
    start_time = time()
    nn = NeuralNetwork(
            train_data,
            test_data,
            HIDDEN_NODES,
            OUTPUT_NUMBER,
            ETA,
            VALIDATION_DATA_PROPORTION,
            VALIDATION_THRESHOLD)
    nn.learn(iterations)
    elapsed_time = time() - start_time

    run_time_results[i] = elapsed_time
    run_success_results[i] = nn.get_success_rate()

for i in range(0, len(run_time_results)):
        print('{},{}'.format(run_time_results[i], run_success_results[i]))
