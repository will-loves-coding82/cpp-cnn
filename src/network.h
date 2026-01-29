#ifndef NETWORK_H_
#define NETWORK_H_

#include "./layer.h"
#include "./loss.h"
#include "./optimizer.h"
#include <vector>

/**
 * Class that configures the Neural Network
 */
class Network {
    private:
        std::vector<Layer *> layers;
        Loss *loss;

    public:
        // Number of correct predictions for a forward pass
        int numCorrect;

        // constructor
        Network() : loss(NULL){};

        // destructor
        ~Network() {
            for (int i = 0; i < layers.size(); i++) {
                delete layers[i];
            }
            if (loss) {
                delete loss;
            }
        }

        void add_layer(Layer *layer) { layers.push_back(layer); };
        void add_loss(Loss *loss_in) { loss = loss_in; };

        float get_loss() { return loss->output(); };
        float get_accuracy() { return loss->output(); };

        int get_num_correct() { return loss->get_num_correct(); };
        int get_num_layers() { return layers.size(); };

        void forward(const Matrix &input);
        void backward(const Matrix &pred, const Matrix &target);
        void update(Optimizer &opt);

        const Matrix& output() { return layers.back()->output(); };
};

#endif