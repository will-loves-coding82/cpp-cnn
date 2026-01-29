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
    int batch_size = bottom.cols();
    top.resize(dim_out, batch_size);

    top = (weight.transpose() * bottom);
    top.colwise() += bias;
}

void FC::backward(const Matrix &bottom, const Matrix &grad_top) {
    int batch_size = bottom.cols();
    grad_weight = bottom * grad_top.transpose();
    grad_bias = grad_top.rowwise().sum();

    grad_bottom.resize(dim_in, batch_size);
    grad_bottom = weight * grad_top;
}

void FC::update(Optimizer &opt)
{

    std::cout << "Updating fully connected" << std::endl;
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);

    grad_weight.setZero();
    grad_bias.setZero();
}
