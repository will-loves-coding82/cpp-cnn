#include "./fc.h"
#include <iostream>

FC::FC(int input_dim, int output_dim) {
    dim_in = input_dim;
    dim_out = output_dim;

    weight.resize(dim_in, dim_out);
    bias.resize(dim_out);

    grad_weight.resize(dim_in, dim_out);
    grad_bias.resize(dim_out);

    // Initialize Weights and Bias
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
}

void FC::forward(const Matrix &bottom) {
    std::cout << "computing FC forward" << std::endl;
    std::cout << "bottom rows: " << bottom.rows() << ", cols: " << bottom.cols() << std::endl;
    std::cout << "weight rows: " << weight.rows() << ", cols: " << weight.cols() << std::endl;
    std::cout << "weight.transpose() rows: " << weight.transpose().rows() << ", cols: " << weight.transpose().cols() << std::endl;

    int batch_size = bottom.cols();
    top.resize(dim_out, batch_size);

    top = (weight.transpose() * bottom);
    top.colwise() += bias;
}

void FC::backward(const Matrix &bottom, const Matrix &grad_top) {
    std::cout << "computing FC backward" << std::endl;
    std::cout << "bottom rows: " << bottom.rows() << ", cols: " << bottom.cols() << std::endl;
    std::cout << "grad_top rows: " << grad_top.rows() << ", cols: " << grad_top.cols() << std::endl;

    int batch_size = bottom.cols();
    grad_weight += bottom * grad_top.transpose();
    grad_bias += grad_top.rowwise().sum();

    grad_bottom.resize(dim_in, batch_size);
    grad_bottom = weight * grad_top;
}