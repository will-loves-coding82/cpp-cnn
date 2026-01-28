#include "./sgd.h"

void SGD::update(Vector::AlignedMapType &param, Vector::ConstAlignedMapType &grad) {
    param -= -learning_rate * grad;
};