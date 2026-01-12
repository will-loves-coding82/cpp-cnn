#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include "./network.h"
#include "./layers/convolution.h"
#include "./layers/relu.h"
#include "./layers/fc.h"
#include "./loss.h"
#include "./loss/softmax_loss.h"
#include "./opt/sgd.h"
#include "./mnist/mnist.h"

int main() {

    Matrix train_images, train_labels;
    Matrix test_images, test_labels;
    MNIST mnist("./data/");
    mnist.read();

    int train_size = mnist.train_data.size();
    int test_size = mnist.test_data.size();

    std::cout << "Training set size: " << train_size << std::endl;
    std::cout << "Test set size: " << test_size << std::endl;
    
    // Set up network layers
    std::cout << "Setting up network layers" << std::endl;
    Network cnn;
    Layer *conv1 = new Convolution(28, 28, 5, 5, 1, 64, 1);
    Layer *relu1 = new ReLU();
    Layer *conv2 = new Convolution(24,24, 5, 5, 64, 64, 1);
    Layer *relu2 = new ReLU();
    Layer *fc1 = new FC(conv2->output_dim(), 128);
    Layer *relu3 = new ReLU();
    Layer *fc2 = new FC(fc1->output_dim(), 10);

    cnn.add_layer(conv1);
    cnn.add_layer(relu1);
    cnn.add_layer(conv2);
    cnn.add_layer(relu2);
    cnn.add_layer(fc1);
    cnn.add_layer(relu3);
    cnn.add_layer(fc2);

    Loss* loss = new SoftMaxLoss();
    Optimizer* opt = new SGD(0.1);
    cnn.add_loss(loss);

    int epochs = 10;
    int batch_size = 64;
    int num_batches = train_size / batch_size;

    std::cout << "\nTraining parameters:" << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;

    // Perform training loop
    // Make sure to optimize using SGD
    // Log loss and accuracy at end of each epoch
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;

        float epoch_loss = 0;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            Matrix batch_data = mnist.train_data.block(
                0,                          // Start from row 0 (all features)
                batch * batch_size,         // Start column
                mnist.train_data.rows(),    // All rows (all 784 features)
                batch_size);

            Matrix batch_labels = mnist.train_labels.block(
                0,                       // Start from row 0 (all features)
                batch * batch_size,      // Start column
                mnist.train_labels.rows(), // All rows (all 784 features)
                batch_size);

            std::cout << "Computing foward pass" << std::endl;
            std::cout << "CNN input size: " << batch_data.size() << std::endl;

            cnn.forward(batch_data);

            std::cout << "Evaluating loss" << std::endl;

            std::cout << "CNN output size: " << cnn.output().size() << std::endl;
            std::cout << "Batch label size: " << batch_labels.size() << std::endl;
            loss->evaluate(cnn.output(), batch_labels);

            std::cout << "Computing backward pass" << std::endl;
            cnn.backward(batch_data, loss->back_gradient());
            cnn.update(*opt);

            if ((batch + 1) % 100 == 0) {
                std::cout << "  Batch " << (batch + 1) << "/" << num_batches
                          << "\r" << std::flush;
            }
        }

        epoch_loss /= num_batches;
        float accuracy = 100.0f * correct / (num_batches * batch_size);

        std::cout << "  Loss: " << epoch_loss
                  << " - Accuracy: " << accuracy << "%" << std::endl;
    }
}