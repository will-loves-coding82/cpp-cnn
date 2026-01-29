#include "./softmax.h"
#include <iostream>

void Softmax::forward(const Matrix& bottom) {
    top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
    RowVector z_exp_sum = top.colwise().sum(); // \sum{ exp(z) }
    top.array().rowwise() /= z_exp_sum;
}


void Softmax::backward(const Matrix& bottom, const Matrix& grad_top) {
    RowVector sum = top.cwiseProduct(grad_top).colwise().sum();
    grad_bottom = top.array().cwiseProduct(grad_top.array().rowwise() - sum);
}