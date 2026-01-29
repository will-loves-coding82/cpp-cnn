#include <chrono>
#include <iostream>
#include <iomanip>
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

    // Redirect std::cout logs to output text file
    std::ofstream outputFile("output.txt");
    if(!outputFile.is_open()) {
        std::cerr << "Could not open output.txt file" << std::endl;
        return 1;
    }
    std::streambuf *streamBuffer = std::cout.rdbuf();
    std::cout.rdbuf(outputFile.rdbuf());

    Matrix train_images, train_labels;
    Matrix test_images, test_labels;
    MNIST mnist("./data/");
    mnist.read();

    int total_train_samples = mnist.train_data.cols();
    int total_test_samples = mnist.test_data.cols();

    // Select a small subset of training and test data for quick experimentation
    int train_size = std::min(1000, total_train_samples);
    int test_size = std::min(32, total_test_samples);

    std::cout << "Training set size: " << train_size << std::endl;
    std::cout << "Test set size: " << test_size << std::endl;

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
    float learning_rate = 0.005f;
    Optimizer *opt = new SGD(learning_rate);
    cnn.add_loss(loss);

    int epochs = 10;
    int batch_size = 32;
    int num_batches = train_size / batch_size;

    std::cout << "\nTraining parameters:" << std::endl;
    std::cout << "  Num layers: " << cnn.get_num_layers() << std::endl;
    std::cout << "  Epochs: " << epochs << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;

    std::cout << "\nMnist train data dimensions: (" <<  mnist.train_data.rows() << " rows, " << mnist.train_data.cols() << " cols)" <<  std::endl;
    std::cout << "Mnist train label dimensions: (" << mnist.train_labels.rows() << " rows, " << mnist.train_labels.cols() << " cols)" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        int epoch_correct = 0;
        float epoch_loss = 0;
        float epoch_accuracy = 0;

        for (int batch = 0; batch < num_batches; batch++) {
            int batch_start = batch * batch_size;
            int current_batch_size = std::min(batch_size, train_size - batch_start);

            // Extract columns [batch_start ... batch_start + current_batch_size)
            Matrix batch_data = train_images.block( // Use normalized data
                0, batch_start, 784, current_batch_size);

            Matrix batch_labels_raw = train_labels.block(
                0, batch_start, 1, current_batch_size);
                
            Matrix labels_encoded = one_hot_encode(batch_labels_raw, 10);

            cnn.forward(batch_data);       
            cnn.backward(batch_data, labels_encoded);
            cnn.update(*opt);

            epoch_loss += cnn.get_loss();
            epoch_correct += cnn.get_num_correct();
        }

        epoch_loss /= num_batches;
        epoch_accuracy = (float(epoch_correct) / train_size) * 100.0f;

        // Log the networks loss and accuracy at the end of each epoch
        std::cout << "Epoch training loss: " << epoch_loss << std::endl;
        std::cout << std::setprecision(4) << "Epoch training accuracy: " << epoch_accuracy << "%" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;
    }

    // Then perform predictions on the test dataset and determine the performance
    std::cout << "\n--- Test Set Evaluation ---" << std::endl;
    cnn.forward(test_images);
    loss->evaluate(cnn.output(), one_hot_encode(test_labels, 10));
    std::cout << "Test loss: " << cnn.get_loss() << std::endl;
    std::cout << "Test accuracy: " << (float(cnn.get_num_correct()) / test_size) * 100.f << "%" << std::endl;

    // Close output text file and restore std::cout buffer
    outputFile.close();
    std::cout.rdbuf(streamBuffer);
}