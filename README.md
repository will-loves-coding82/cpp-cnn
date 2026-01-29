## Overview
A convolutional neural network implementation inspired by the mini-DNN framework. This project was developed to deepen my understanding of the mathematical foundations underlying neural network operations and gradient computation during backpropagation. The codebase is primarily implemented in C++ and provides abstractions comparable to high-level Python frameworks such as Keras layers. Future development will focus on optimizing performance through GPU acceleration.

## Code Organization

```bin/```
This folder contains a C++ binary that will run the training process and perform predictions on a small test dataset. Metrics such as loss and accuracy will be provided at the end of each epoch to observe how the model changes over time. Logs will be written to an `output.txt` file for execution tracing.

```data/```
This folder contains a dataset of 10k MNIST images and labels. 

```src/```
This folder contains the code required to build the convolutional neural network
 
 - `Eigen`: C++ library that contains data structures for linear algebra operations
 - `layers`: Contains the header and cpp files for convolution, fully connected, reLU, and softmax layers
 - `loss`: Contains the cross entropy code to evaluate the loss during training
 - `mnist`: Helper class from a 3rd party github repository that can load MNIST data
 - `opt`: Contains the code for the Stochastic Gradient Descent, responsible for updating the networks weights at a specific layer.
 - `main.cpp`: Builds the CNN using 2
 - `network.cpp`

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
There should be some rudimentary scripts for building your project's code in an automatic fashion.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

