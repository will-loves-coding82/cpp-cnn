#ifndef SOFTMAX_H_
#define SOFTMAX_H_
#include "../utils.h"
#include "../layer.h"

class Softmax : public Layer {
    public:
        void forward(const Matrix& bottom);
        void backward(const Matrix &bottom, const Matrix &grad_top);
};

#endif