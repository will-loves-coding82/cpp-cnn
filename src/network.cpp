#include "./network.h"

void Network::forward (const Matrix &input) {
    if (layers.empty()) {
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
    loss->evaluate(input, target);

    // Propogate through previous layers
    for (int i = num_layers - 2; i > 0; i--) {
        // Calulcate gradient for eachl
        layers[i]->backward(layers[i-1]->output(), layers[i + 1]->back_gradient());
    }
    
}

void Network::update(Optimizer &opt) {
    int num_layers = layers.size();

    for (int i = num_layers; i > 0; i--)
    {
        // Update weights for each layer
        layers[i]->update(opt);
    }
}