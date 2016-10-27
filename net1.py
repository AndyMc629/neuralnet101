# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:10:30 2016

Based on Sirajology's YouTube video:
https://www.youtube.com/watch?v=h3l4qz76JhQ


The program creates an neural network that simulates
the exclusive OR function with two inputs and one output.

@author: andrew
"""
import numpy as np

# sigmoid function


def nonlin(x, deriv=False):
    if(deriv):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data. 4 examples with 3 input neurons each
# third column is for bias.
X = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

# output data, 4 examples with 1 output neuron each
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# initialise the weights to random values. syn0 = matrix of weights behind
# the input and hidden layer, 3x4 matrix.
# It is a 3x4 matrix because there are two input weights plus a bias term (=3)
# and four nodes in the hidden layer (=4). syn1 are the weights between the
# hidden layer and the output layer. It is a 4x1 matrix because there are
# 4 nodes in the hidden layer and one output. Note that there is no bias term
# feeding the output layer in this example. The weights are initially generated
# randomly because optimization tends not to work well when all the weights
# start at the same value. Note that neither of the neural networks shown
# in the video describe the example.

# synapses. 3 layer net = 2 synapse matrices.
# 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn0 = 2*np.random.random((3, 4)) - 1
# 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the
# hidden layer.
syn1 = 2*np.random.random((4, 1)) - 1

# This is the main training loop. The output shows the evolution of the error
# between the model and desired. The error steadily decreases.

# training step
for j in range(600000):

    # Calculate forward through the network.
    # i.e multiply each layer by it's synapse matrix.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Back propagation of errors using the chain rule.
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps
        print "Error: " + str(np.mean(np.abs(l2_error)))

    l2_delta = l2_error*nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights using gradient descent (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "Output after training"
print l2
