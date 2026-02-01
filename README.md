## Overview
A convolutional neural network implementation inspired by the mini-DNN framework. This project was developed to deepen my understanding of the mathematical foundations underlying neural network operations and gradient computation during backpropagation. The codebase is primarily implemented in C++ and provides abstractions comparable to high-level Python frameworks such as Keras layers. Claude AI was utilized to help troubleshoot syntax and core logic throughout the development process. Future development will focus on optimizing performance through GPU acceleration. 

<hr/>

## Code Organization

```Makefile or CMAkeLists.txt/```
Helps with building the project's code in an automatic fashion.


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
 - `main.cpp`: Builds the CNN and scaffolds the training / test processes
 - `network.cpp`: Defines the class implementation to manage the neural network's layers.

<hr/>


## Results
To train the model, I selected a subset of 1000 MNIST training images and labels. I then performed 10 epochs (i.e. 10 complete passes through the training set) while observing the average loss and accuracy at the end of each epoch. 

The model performed well but soon experienced overfitting with an average accuracy of 99.1% by the end of the training process. Going forward, the model could benefit from several modifications such as validation datasets, regularization, dropout layers, and max pooling in order to prevent overfitting.
