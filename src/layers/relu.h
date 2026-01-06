#ifndef RELU_H_
#define RELU_H_

#include "../layer.h"

class ReLU : public Layer {
    public:
        void forward(Matrix &bottom);
        void backward(Matrix &bottom, Matrix &grad_top);
};

#endif