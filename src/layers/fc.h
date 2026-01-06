#include "../layer.h"
#include <Eigen/Dense>

class FC : public Layer {
    protected:
        Matrix weight;
        Vector bias;

        Matrix grad_weight;
        Vector grad_bias;

        // Input and output dimensions
        int dim_in, dim_out;

        void init();

    public:
        FC(int input_dim, int output_dim) { init(); };
        void forward(Matrix &bottom);
        void backward(Matrix &bottom, Matrix &grad_top);
};