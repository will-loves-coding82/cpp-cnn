#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include "./network.h"
#include "./convolution.h"
#include "./relu.h"
#include "./fc.h"
#include "./optimizer.h"
#include "./loss.h"
#include "mnist_loader.h"
#include "./loss/softmax_loss.h"

int main() {

    // Set up network layers
    Network cnn;
    Layer *conv1 = new Convolution();
    Layer *relu1 = new ReLU();
    Layer *conv2 = new Convolution();
    Layer *relu2 = new ReLU();
    Layer *fc1 = new FC();
    Layer *relu3 = new ReLU();
    Layer *fc2 = new FC();

    cnn.add_layer(conv1);
    cnn.add_layer(relu1);
    cnn.add_layer(conv2);
    cnn.add_layer(relu2);
    cnn.add_layer(fc1);
    cnn.add_layer(relu3);
    cnn.add_layer(fc2);

    Loss *loss = new SoftmaxLoss;
    cnn.add_loss(loss);

    SGD opt(0.1);

    int epochs = 10;
    int batch_size = 64;
    int num_batches = epochs / batch_size;

    std::cout << "\nTraining parameters:" << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;
    std::cout << "  Learning rate: " << optimizer.learning_rate << std::endl;

    // Perform training loop
    // Make sure to optimize using SGD
    // Log loss and accuracy at end of each epoch
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float epoch_loss = 0;
        int correct = 0;

    }
    // Save network weights in file

    // Evaluate performance on test dataset
    }