import numpy as np
import argparse
from matplotlib import pyplot as plt
from time import time
from neural_network import NeuralNetwork
import pickle
import os

HIDDEN_NODES = 20
OUTPUT_NUMBER = 10
ETA = 0.3
ITERATIONS = 10000
VALIDATION_DATA_PROPORTION = 0.2
VALIDATION_THRESHOLD = 0.001

CHANGE_HIDDEN_NODES = 1
CHANGE_ITERATIONS =2
CHANGE_LEARNING_RATE = 3
REGULAR_RUN = 4

np.random.seed(int(time()))

parser = argparse.ArgumentParser(description='Test the neural network')
parser.add_argument('--test', type=int, required=True, help='''
1. Perform test changing the number of hidden nodes
2. Perform test changing the number of iterations
3. Perform test changing the learing rate
4. Perform just one run''' )

parser.add_argument('--training_file', type=str, required=True,
        help='File containing the training cases for the neural network')
parser.add_argument('--testing_file', type=str, required=True,
        help='File containing the testing cases for the neural network')
parser.add_argument('--runs', type=int, help='Number of runs for each test')
parser.set_defaults(runs=1)

parser.add_argument('--min_nodes', type=int,
        help='Minimum number of hidden nodes')
parser.add_argument('--max_nodes', type=int,
        help='Maximum number of hidden nodes')
parser.add_argument('--nodes_step', type=int,
        help='Increment for the number of hidden nodes')
parser.set_defaults(nodes_step=1)

parser.add_argument('--min_iterations', type=int,
        help='Minimum number of iterations')
parser.add_argument('--max_iterations', type=int,
        help='Maximum number of iterations')
parser.add_argument('--iterations_step', type=int,
        help='Increment for the number of iterations')
parser.set_defaults(iterations_step=1)

parser.add_argument('--min_learning_rate', type=int,
        help='Minimum learning rate')
parser.add_argument('--max_learning_rate', type=int,
        help='Maximum learning rate')
parser.add_argument('--learning_rate_step', type=int,
        help='Increment for the learning rate')
parser.set_defaults(learning_rate_step=1)

# Test 1: Changing the number of hidden nodes
def change_hidden_nodes(train_data, test_data, min_hidden_nodes,
        max_hidden_nodes, hidden_nodes_step, runs):

    hidden_nodes_range = np.arange(min_hidden_nodes, max_hidden_nodes, hidden_nodes_step)
    time_results = np.zeros(len(hidden_nodes_range))
    success_results = np.zeros(len(hidden_nodes_range))
    for hidden_nodes in hidden_nodes_range:
        run_time_results = np.zeros(runs)
        run_success_results = np.zeros(runs)
        for i in range(0, runs):
            start_time = time()
            nn = NeuralNetwork(
                    train_data,
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

    output_results(time_results, success_results, hidden_nodes_range)
    plot(time_results, success_results, hidden_nodes_range, 'Hidden Nodes')

# Test 2: Changing Iterations
def change_iterations(train_data, test_data, min_iterations, max_iterations,
        iterations_step, runs):

    iterations_range = range(min_iterations, max_iterations, iterations_step)
    time_results = np.zeros(len(iterations_range))
    success_results = np.zeros(len(iterations_range))
    for iterations in iterations_range:
        run_time_results = np.zeros(runs)
        run_success_results = np.zeros(runs)
        for i in range(0, runs):
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
        time_results[i] = np.mean(run_time_results)
        success_results[i] = np.mean(run_success_results)

    output_results(time_results, success_results, iterations_range)
    plot(time_results, success_results, iterations_range, 'Iterations')

# Test 3: Changing ETA
def change_learning_rate(train_data, test_data, min_learning_rate,
        max_learning_rate, learning_rate_step, runs):

    eta_range = np.arange(min_learning_rate, max_learning_rate, learning_rate_step)
    time_results = np.zeros(len(eta_range))
    success_results = np.zeros(len(eta_range))
    for eta in eta_range:
        run_time_results = np.zeros(runs)
        run_success_results = np.zeros(runs)
        for i in range(0, runs):
            start_time = time()
            nn = NeuralNetwork(
                    train_data,
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

    output_results(time_results, success_results, eta_range)
    plot(time_results, success_results, eta_range, 'ETA')

# Test 4: Regular Run
def regular_run(train_data, test_data, runs):
    run_time_results = np.zeros(runs)
    run_time_results.shape = (runs, 1)
    run_success_results = np.zeros(runs)
    run_success_results.shape = (runs, 1)

    for i in range(0, runs):
        start_time = time()
        nn = NeuralNetwork(
                train_data,
                test_data,
                HIDDEN_NODES,
                OUTPUT_NUMBER,
                ETA,
                VALIDATION_DATA_PROPORTION,
                VALIDATION_THRESHOLD)
        nn.learn(ITERATIONS)
        elapsed_time = time() - start_time
        run_time_results[i] = elapsed_time
        run_success_results[i] = nn.get_success_rate()

    print('Average time: {}'.format(np.mean(run_time_results)))
    print('Average success rate: {}'.format(np.mean(run_success_results)))


def plot(time_results, success_results, x_axis, xlabel):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(time_results, x_axis, 'bo-')
    plt.ylabel('Execution Time (s)')
    plt.xlabel(xlabel)

    plt.subplot(212)
    plt.plot(success_results, x_axis, 'rx-')
    plt.ylabel('Success Rate (%)')
    plt.xlabel(xlabel)
    file_name ='output/{}-{}-plot.png'.format(args.test, time())
    plt.savefig(file_name, format='png')

def output_results(time_results, success_results, data_range):

    combined_results = np.array((time_results, success_results, data_range))
    file_name = 'output/{}-{}-test_results.csv'.format(args.test, time())
    np.savetxt(file_name, combined_results, delimiter=',')

if __name__ == '__main__':
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

    read_elapsed_time = time() - read_start_time

    print("Read time {}".format(read_elapsed_time))

    if args.test == CHANGE_HIDDEN_NODES:
        change_hidden_nodes(train_data, test_data, args.min_nodes,
                args.max_nodes, args.nodes_step, args.runs)
    elif args.test == CHANGE_ITERATIONS:
        change_iterations(train_data, test_data, args.min_iterations,
                args.max_iterations, args.iterations_step, args.runs)
    elif args.test == CHANGE_LEARNING_RATE:
        change_learning_rate(train_data, test_data,  args.min_learning_rate,
                args.max_learning_rate, args.learning_rate_step, args.runs)
    elif args.test == REGULAR_RUN:
        regular_run(train_data, test_data, args.runs)
