#include "./convolution.h"

Convolution::Convolution(int height_in, int width_in, int kernel_width, int kernel_height, int channel_in, int channel_out, int stride) {

    weight.resize(channel_in * kernel_height * kernel_width, channel_out);
    bias.resize(channel_out);

    grad_weight.resize(channel_in * kernel_height * kernel_width, channel_out);
    grad_bias.resize(channel_out);

    // Initialize Weights and Bias
};

