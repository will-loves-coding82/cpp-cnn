#include "./fc.h"


FC::FC(int input_dim, int output_dim) {
    dim_in = input_dim;
    dim_out = output_dim;

    weight.resize(dim_in, dim_out);
    bias.resize(dim_out);

    grad_weight.resize(dim_in, dim_out);
    grad_bias.resize(dim_out);

    // Initialize Weights and Bias
}

void FC::forward(Matrix &bottom) {
    int batch_size = bottom.cols();
    top.resize(dim_out, batch_size);

    top = (weight.transpose() * bottom);
    top.colwise() += bias;
}

void FC::backward(Matrix &bottom, Matrix &grad_top) {
    int batch_size = bottom.cols();

    grad_weight += bottom.transpose() * grad_top;
    grad_bias += grad_top.rowwise().sum();

    grad_bottom.resize(dim_in, batch_size);
    grad_bottom = weight * grad_top;
}