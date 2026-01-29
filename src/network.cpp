#include "./network.h"
#include <iostream>

void Network::forward (const Matrix &input) {
    if (layers.empty()) {
        std::cout << "layers are empty" << std::endl;
        return;
    }
    layers[0]->forward(input);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i - 1]->output());
    }
}

void Network::backward(const Matrix &input, const Matrix &target) {
    int num_layers = layers.size();
    if (num_layers == 0) {
        return;
    }

    // Compute the loss and the softmax gradient w.r.t input
    loss->evaluate(layers[num_layers - 1]->output(), target);

    // Propogate through previous layers
    layers[num_layers- 1]->backward(layers[num_layers - 2]->output(),loss->back_gradient());
    for (int i = num_layers - 2; i > 0; i--) {
        layers[i]->backward(layers[i-1]->output(), layers[i + 1]->back_gradient());
    }

    // Update the first layer w.r.t the input
    layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(Optimizer &opt) {
    int num_layers = layers.size();
    for (int i = 0; i < num_layers; i++){
        // Update weights for each layer
        layers[i]->update(opt);
    }
}