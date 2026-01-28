#include <chrono>
#include <iostream>
#include <string>
#include <tuple>

#include "./network.h"
#include "./layers/convolution.h"
#include "./layers/relu.h"
#include "./layers/fc.h"
#include "./layers/softmax.h"
#include "./loss/cross_entropy.h"
#include "./opt/sgd.h"
#include "./mnist/mnist.h"
#include <algorithm>

int main() {

    Matrix train_images, train_labels;
    Matrix test_images, test_labels;
    MNIST mnist("./data/");
    mnist.read();

    int total_train_samples = mnist.train_data.cols();
    int total_test_samples = mnist.test_data.cols();

    int train_size = std::min(100, total_train_samples);
    int test_size = std::min(30, total_test_samples);

    std::cout << "Training set size: " << train_size << std::endl;
    std::cout << "Test set size: " << test_size << std::endl;

    // Select a small subset of training and test data for fast experimentation
    train_images = mnist.train_data.block(0, 0, 784, train_size);
    train_labels = mnist.train_labels.block(0, 0, 1, train_size);

    test_images = mnist.test_data.block(0, 0, 784, test_size);
    test_labels = mnist.test_labels.block(0, 0, 1, test_size);

    // Set up network layers
    std::cout << "Setting up network layers" << std::endl;
    Network cnn;
    Layer *conv1 = new Convolution(28, 28, 5, 5, 1, 64, 1);
    Layer *relu1 = new ReLU();
    Layer *conv2 = new Convolution(24,24, 5, 5, 64, 64, 1);
    Layer *relu2 = new ReLU();
    Layer *fc1 = new FC(conv2->output_dim(), 64);
    Layer *relu3 = new ReLU();
    Layer *fc2 = new FC(fc1->output_dim(), 10);
    Layer *softmax = new Softmax();
    
    cnn.add_layer(conv1);
    cnn.add_layer(relu1);
    cnn.add_layer(conv2);
    cnn.add_layer(relu2);
    cnn.add_layer(fc1);
    cnn.add_layer(relu3);
    cnn.add_layer(fc2);
    cnn.add_layer(softmax);

    Loss* loss = new CrossEntropy();
    Optimizer* opt = new SGD(0.1);
    cnn.add_loss(loss);

    int epochs = 1;
    int batch_size = 32;
    int num_batches = train_size / batch_size;

    std::cout << "\nTraining parameters:" << std::endl;
    std::cout << "  Num layers: " << cnn.get_num_layers() << std::endl;

    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;

    // 784 rows * 60000 columns
    std::cout << "\nMnist train data dimensions: (" <<  mnist.train_data.rows() << " rows, " << mnist.train_data.cols() << " cols)" <<  std::endl;
    
    // 1 row * 60000 columns
    std::cout << "\nMnist train label dimensions: (" << mnist.train_labels.rows() << " rows, " << mnist.train_labels.cols() << " cols)" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;

        float epoch_loss = 0;
        int correct = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * batch_size;
            int current_batch_size = std::min(batch_size, train_size - batch_start);

            // Extract columns [batch_start ... batch_start + current_batch_size)
            Matrix batch_data = mnist.train_data.block(
                0,           // Start from row 0
                batch_start, // Start column (sample index)
                784,         // All 784 features
                current_batch_size);

            Matrix batch_labels_raw = mnist.train_labels.block(
                0,           // Start from row 0
                batch_start, // Start column (sample index)
                1,           // Only 1 row available
                current_batch_size);

            Matrix labels_encoded = one_hot_encode(batch_labels_raw, 10);

            std::cout << "CNN batch input size: " << batch_data.cols() << std::endl;
            std::cout << "Encoded labels size: " << labels_encoded.cols() << std::endl;

            cnn.forward(batch_data);
            std::cout << "CNN output size: " << cnn.output().size() << std::endl;
        
            std::cout << "Computing backward pass" << std::endl;
            cnn.backward(batch_data, labels_encoded);

            std::cout << "Updating weights with gradients" << std::endl;
            cnn.update(*opt);

            if ((batch_size + 1) % 100 == 0) {
                std::cout << "  Batch " << (batch + 1) << "/" << num_batches
                          << "\r" << std::flush;
            }
        }
    }
}