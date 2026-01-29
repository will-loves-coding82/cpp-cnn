#include "./sgd.h"

void SGD::update(Vector::AlignedMapType &param, Vector::ConstAlignedMapType &grad) {
    param -= learning_rate * grad.cwiseMax(-1.0f).cwiseMin(1.0f);;
};