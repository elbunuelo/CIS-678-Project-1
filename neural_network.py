#! /usr/local/bin/python3

import numpy as np


class NeuralNetwork:
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2

    def __init__(self, all_train_data, test_data,
            number_of_hidden_nodes, number_of_outputs, eta,
            validation_data_proportion, validation_threshold):

        self.eta = eta
        self.number_of_outputs = number_of_outputs
        self.all_train_data = all_train_data
        self.test_data = test_data

        all_data_length = self.all_train_data.shape[0]
        number_of_inputs = self.all_train_data.shape[1] - 1

        # Hidden Neurons - Dimensions 1 x h+1
        self.hidden_neurons = np.full(number_of_hidden_nodes + 1, -1.0)
        self.hidden_neurons.shape = (1, self.hidden_neurons.shape[0])

        # Input Neurons - Dimensions 1 x n+1
        self.input_neurons = np.full(number_of_inputs + 1, -1.0)
        self.input_neurons.shape = (1, self.input_neurons.shape[0])

        # Input Layer -> Hidden Layer - Dimensions n+1 x h
        random_initializer = np.random.rand( number_of_inputs + 1, number_of_hidden_nodes)
        self.input_hidden_weights = (random_initializer - 1/2) * 2/np.sqrt(number_of_inputs)

        # Hidden Layer ->  Output weights - Dimensions h+1 x o
        random_initializer = np.random.rand(number_of_hidden_nodes + 1, number_of_outputs)
        self.hidden_output_weights = (random_initializer - 1/2) * 2/np.sqrt(number_of_inputs)

        # Create errors arrays
        self.validation_data_proportion = validation_data_proportion
        self.validation_dataset_length = int(all_data_length * (validation_data_proportion))
        self.training_dataset_length = all_data_length - self.validation_dataset_length
        self.training_errors = np.zeros((self.training_dataset_length, number_of_outputs))
        self.validation_errors = np.zeros((self.validation_dataset_length, number_of_outputs))

        self.test_results = np.zeros(self.test_data.shape[0], dtype=bool)

        self.validation_threshold = validation_threshold

    def get_targets(self, element=None, label=None):
            #Target
            if not label:
                label = int(element[0])

            #Dimensions o x 1
            targets = np.zeros(self.number_of_outputs)
            targets[label] = 1

            return targets

    def forward(self):
            #Forward inputs to hidden layer
            sigma = np.dot(self.input_neurons, self.input_hidden_weights)

            # Dimensions 1 x h
            hidden_output = 1/(1 + np.exp(-sigma))

            # Update hidden neurons
            self.hidden_neurons[:, 1:] = hidden_output

            #Forward hidden layer to output
            sigma = np.dot(self.hidden_neurons, self.hidden_output_weights)

            # Dimensions 1 x o
            self.outputs = 1/(1 + np.exp(-sigma))

    def back_propagate(self, targets):
            # Output Error Ey = y(1-y)(t-y)
            # Dimensions 1 x o
            output_errors = self.outputs * (1 - self.outputs) * (targets - self.outputs)

            # Hidden Error Eh = h(1-h)(Why (dot) Ey)
            # Dimensions 1 x h
            non_bias_hidden_neurons = self.hidden_neurons[:, 1:]
            non_bias_hidden_output_weights = self.hidden_output_weights[1:, :]
            hidden_errors = (
                    non_bias_hidden_neurons *
                    (1 - non_bias_hidden_neurons) *
                    np.dot(output_errors, non_bias_hidden_output_weights.T))

            # Update weights w = w + eta * E * input
            self.input_hidden_weights += self.eta * (self.input_neurons.T * hidden_errors)
            self.hidden_output_weights += self.eta * (self.hidden_neurons.T * output_errors)

    def check_output(self, test_element, i):
        result = self.outputs.argmax()
        expected = test_element[0]

        self.test_results[i] = (result == expected)


    def run(self, data, mode):
        for i, element in enumerate(data):
            targets = self.get_targets(element)

            #Update input neurons
            self.input_neurons[:, 1:] = element[1:]

            self.forward()

            if mode == self.TRAINING:
                self.training_errors[i, :] = np.square(self.outputs - targets)
                self.back_propagate(targets)
            elif mode == self.VALIDATION:
                self.validation_errors[i, :] = np.square(self.outputs - targets)
            elif mode == self.TESTING:
                self.check_output(element, i)


    def get_iteration_data(self):
        np.random.shuffle(self.all_train_data)
        validation_data = self.all_train_data[self.training_dataset_length:,:]
        train_data = self.all_train_data[:self.training_dataset_length, :]

        return {
                'training': train_data,
                'validation': validation_data
                }

    def check_validation_errors(self, errors, i):
        if (i < 2):
            return False

        threshold = self.validation_threshold

        current = errors[i]
        previous = errors[i-1]
        two_before = errors[i-2]
        return (previous - current) <= threshold and (two_before - previous) <= threshold


    def learn(self, iterations):
        sum_square_errors_training = np.zeros(iterations)
        sum_square_errors_validation = np.zeros(iterations)
        for i in range(0, iterations):
            data = self.get_iteration_data()
            training_data = data['training']
            validation_data = data['validation']

            self.run(training_data, self.TRAINING)
            sum_square_errors_training[i] = 1/2 * np.sum(self.training_errors)
            self.run(validation_data, self.VALIDATION)
            sum_square_errors_validation[i] = 1/2 * np.sum(self.validation_errors)

            if self.check_validation_errors(sum_square_errors_validation, i):
                break

        self.run(self.test_data, self.TESTING)

    def get_success_rate(self):
        return np.sum(self.test_results)/self.test_results.shape[0]


