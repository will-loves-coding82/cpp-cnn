#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

#include "../layer.h"

// Valid padding convolution for simplicity
class Convolution : public Layer {
    protected:
        Matrix weight;
        Vector bias;

        Matrix grad_weight;
        Vector grad_bias;

        // Stores the im2col representations for each image in a batch
        std::vector<Matrix> data_cols;

        int dim_in, dim_out;
        int height_in, height_out;
        int width_in, width_out;
        int kernel_width, kernel_height;
        int channel_in, channel_out;
        int stride;


        void init();

    public:
        Convolution(int h_in, int w_in, int k_w, int k_h, int c_in, int c_out, int stride = 1);

        void forward(const Matrix &bottom);
        void backward(const Matrix &bottom, const Matrix &grad_top);
        void update(Optimizer &opt);

        int output_dim() { return dim_out; };
        Matrix get_grad_weight() { return grad_weight; };

        void im2col(Vector &img, Matrix &data_col);
        void col2im(Matrix &data_col, Vector &image);
};

#endif