
#ifndef FC_H_
#define FC_H_

#include "../layer.h"
#include "Eigen/Dense"

class FC : public Layer {
    protected:
        Matrix weight;
        Vector bias;

        Matrix grad_weight;
        Vector grad_bias;

        // Input and output dimensions
        int dim_in, dim_out;

    public:
        FC(int input_dim, int output_dim);
        void forward(Matrix &bottom);
        void backward(Matrix &bottom, Matrix &grad_top);
};

#endif