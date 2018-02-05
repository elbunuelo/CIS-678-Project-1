#! /usr/local/bin/python3

import numpy as np
import argparse
from matplotlib import pyplot as plt
from time import time
from neural_network import NeuralNetwork

HIDDEN_NODES = 20
OUTPUT_NUMBER = 10
ETA = 0.3
ITERATIONS = 100000
VALIDATION_DATA_PROPORTION = 0.2
VALIDATION_THRESHOLD = 0.001

CHANGE_HIDDEN_NODES = 1
CHANGE_ITERATIONS =2
CHANGE_LEARNING_RATE = 3


np.random.seed(int(time()))

parser = argparse.ArgumentParser(description='Test the neural network')
parser.add_argument('Test', type=int, help='''
1. Perform test changing the number of hidden nodes
2. Perform test changing the number of iterations
3. Perform test changing the learing rate
    ''' )

# Read all data files
read_start_time = time()
all_train_data = np.genfromtxt('data/normalized_training.csv', delimiter=',')
test_data = np.genfromtxt('data/normalized_testing.csv', delimiter=',')
read_elapsed_time = time() - read_start_time

print("Read time {}".format(read_elapsed_time))

# Test 1: Changing the number of hidden nodes
def change_hidden_nodes(min_hidden_nodes, max_hidden_nodes):
    hidden_nodes_range = range(min_hidden_nodes, max_hidden_nodes)
    time_results = np.zeros(len(hidden_nodes_range))
    success_results = np.zeros(len(hidden_nodes_range))
    for hidden_nodes in hidden_nodes_range:
        run_time_results = np.zeros(10)
        run_success_results = np.zeros(10)
        for i in range(0,10):
            start_time = time()
            nn = NeuralNetwork(
                    all_train_data,
                    test_data,
                    hidden_nodes,
                    OUTPUT_NUMBER,
                    ETA,
                    VALIDATION_DATA_PROPORTION,
                    VALIDATION_THRESHOLD)
            nn.learn(ITERATIONS)
            elapsed_time = time() - start_time
            run_time_results[i] = elapsed_time
            run_success_results[i] = nn.get_success_rate()
        time_results[i] = np.mean(run_time_results)
        success_results[i] = np.mean(run_success_results)

    output_results(time_results, success_results, data_range)
    plot(time_results, success_results, hidden_nodes_range, 'Hidden Nodes')

# Test 2: Changing Iterations
def change_iterations(min_iterations, max_iterations):
    iterations_range = range(min_iterations, max_iterations)
    time_results = np.zeros(len(iterations_range))
    success_results = np.zeros(len(iterations_range))
    for iterations in iterations_range:
        run_time_results = np.zeros(10)
        run_success_results = np.zeros(10)
        for i in range(0, 10):
            start_time = time()
            nn = NeuralNetwork(
                    all_train_data,
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
        time_results[i] = np.mean(run_time_results)
        success_results[i] = np.mean(run_success_results)

    output_results(time_results, success_results, data_range)
    plot(time_results, success_results, iterations_range, 'Iterations')

# Test 3: Changing ETA
def change_learning_rate(min_learning_rate, max_learning_rate, step):
    eta_range = np.arange(min_learning_rate, max_learning_rate, step)
    time_results = np.zeros(len(eta_range))
    success_results = np.zeros(len(eta_range))
    for eta in eta_range:
        run_time_results = np.zeros(10)
        run_success_results = np.zeros(10)
        for i in range(0, 10):
            start_time = time()
            nn = NeuralNetwork(
                    all_train_data,
                    test_data,
                    HIDDEN_NODES,
                    OUTPUT_NUMBER,
                    eta,
                    VALIDATION_DATA_PROPORTION,
                    VALIDATION_THRESHOLD)
            nn.learn(ITERATIONS)
            elapsed_time = time() - start_time
            run_time_results[i] = elapsed_time
            run_success_results[i] = nn.get_success_rate()
        time_results[i] = np.mean(run_time_results)
        success_results[i] = np.mean(run_success_results)

    output_results(time_results, success_results, data_range)
    plot(time_results, success_results, eta_range, 'ETA')

def plot(time_results, success_results, x_axis, xlabel):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time_results, eta_range, 'b-')
    plt.ylabel('Execution Time (s)')
    plt.xlabel(xlabel)

    plt.subplot(212)
    plt.plot(success_results, eta_range, 'r-')
    plt.ylabel('Success Rate (%)')
    plt.xlabel(xlabel)
    plt.savefig('plot.png', format='png')

def output_results(time_results, success_results, data_range):
    combined_results = np.concatenate((time_results, success_results, data_range), axis=1)
    np.savetxt('test_results.csv', combined_results, delimiter=',')

if __name__ == '__main__':
    args = parser.parse_args

    if args.Test == CHANGE_HIDDEN_NODES:
        change_hidden_nodes(1, 200)
    elif args.Test == CHANGE_ITERATIONS:
        change_iterations(1, 151)
    elif args.Test == CHANGE_LEARNING_RATE:
        change_learning_rate(0.1, 0.61, 0.01)

