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

        int height_in, height_out;
        int width_in, width_out;
        int kernel_width, kernel_height;
        int channel_in, channel_out;

        int stride;

        void init();

    public:
        Convolution(
            int h_in, int w_in, int k_w, int k_h, int c_in, int c_out, int stride = 1) :
            height_in(h_in),
            width_in(w_in),
            kernel_width(k_w),
            kernel_height(k_h),
            width_out(w_in - k_w / stride + 1),
            height_out(h_in - k_h / stride + 1)
        { init(); };

        void forward(Matrix &bottom);
        void backward(Matrix &bottom, Matrix &grad_top);

        void im2col(Vector &img, Matrix &data_col);
        void col2im(Matrix &data_col, Vector &image);

        void update(Optimizer &opt) {};
};