#include "./relu.h"

void ReLU::forward(Matrix &bottom){ 
    top = bottom.cwiseMax(0.0);
};

void ReLU::backward(Matrix &bottom, Matrix &grad_top) {
    Matrix positives = (bottom.array() > 0.0).cast<float>();
    grad_bottom = bottom.cwiseProduct(positives);
}