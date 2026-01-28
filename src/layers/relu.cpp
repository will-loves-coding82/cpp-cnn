#include "./relu.h"
#include <iostream>

void ReLU::forward(const Matrix &bottom){
    std::cout << "computing ReLU forward" << std::endl;
    top = bottom.cwiseMax(0.0);
};

void ReLU::backward(const Matrix &bottom, const Matrix &grad_top) {
    Matrix positives = (bottom.array() > 0.0).cast<float>();
    grad_bottom = bottom.cwiseProduct(positives);
}
