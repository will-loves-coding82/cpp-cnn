#include "./convolution.h"

Convolution::Convolution(int height_in, int width_in, int kernel_width, int kernel_height, int channel_in, int channel_out, int stride) {

    weight.resize(channel_in * kernel_height * kernel_width, channel_out);
    bias.resize(channel_out);

    grad_weight.resize(channel_in * kernel_height * kernel_width, channel_out);
    grad_bias.resize(channel_out);

    // Initialize Weights and Bias
    set_normal_random(weight.data(), weight.size(), 0, 0.01);
    set_normal_random(bias.data(), bias.size(), 0, 0.01);
};

void Convolution::forward(Matrix &bottom) {
    int batch_size = bottom.cols();
    for (int i = 0; i < batch_size; i++) {
        // Convert to im2col for GEMM
        Vector image = bottom.col(i);
        Matrix data_col;
        im2col(image, data_col);
        data_cols[i] = data_col;

        Matrix result = data_col * weight;
        result.rowwise() += bias;
        top.col(i) = Eigen::Map<Vector>(result.data(), result.size());
    }
};

// https://github.com/iamhankai/mini-dnn-cpp/blob/master/src/layer/conv.cc
void Convolution::update(Optimizer &opt)
{
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);
}

void Convolution::backward(Matrix &bottom, Matrix &grad_top) {
    int batch_size = bottom.cols();
    for (int i = 0; i < batch_size; i++) {
        Vector grad_top_i = grad_top.col(i);
        Matrix grad_top_i_matrix = Eigen::Map<Matrix>(grad_top_i.data(), height_out * width_out, channel_out);

        grad_weight += data_cols[i].transpose() * grad_top_i_matrix;
        grad_bias += grad_top_i_matrix.colwise().sum().transpose();

        Matrix grad_bottom_i = grad_top_i_matrix * weight.transpose();

        // convert bottom gradients on patches -> gradients on image space
        // This also means we accumulate gradients for each image pixel since
        // there is data redundancy
        Vector image;
        col2im(grad_bottom_i, image);
        grad_bottom.col(i) = image;
    }
};

void Convolution::col2im(Matrix &data_col, Vector &image) {
    image.resize(channel_in * width_in * height_in);
    image.setZero();

    int row = 0;

    // Loop through each row in data_col
    // If stride > 1, not every image pixel will be a nonzero value
    for (int h_out = 0; h_out < height_out; h_out++) {
        for (int w_out = 0; w_out < width_out; w_out++) {
            int col = 0;
            int h_start = h_out * stride;
            int w_start = w_out * stride;

            for (int c = 0; c < channel_in; c++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {
                        int h_in = h_start + kh;
                        int w_in = w_start + kw;
                        int idx = (c * height_in * width_in) + (h_in * width_in) + w_in;
                        image(idx) += data_col(row, col);
                        col++;
                    }
                }
            }
            row++;
        }
    }
}

void Convolution::im2col(Vector &image, Matrix &data_col){
    int num_patches = height_in * height_out;
    int patch_size = kernel_height * kernel_width * channel_in;

    data_col.resize(num_patches, patch_size);

    // row index correponds to an ouput pixel
    int row_idx = 0;

    // Iterate over output positions along each row
    for (int out_h = 0; out_h < height_out; out_h++) {
        for (int out_w = 0; out_w < width_out; out_w++)   {

            // Offset the kernel window by the stride
            int h_start = out_h * stride;
            int w_start = out_w * stride;

            // column index for patch data
            int col_idx = 0;

            // Extract patch data from image vector
            for (int c = 0; c < channel_in; c++) {
                for (int kh = 0; kh < kernel_height; kh++) {
                    for (int kw = 0; kw < kernel_width; kw++) {

                        int h_in = h_start + kh;
                        int w_in = w_start + kw;

                        // 1D-index into image by offseting by the channel + offset by the rows + the actual column
                        int img_idx = (c * height_in * width_in) + (h_in * width_in) + w_in;
                        data_col(row_idx, col_idx) = image(img_idx);
                        col_idx++;
                    }
                }
            }
            row_idx++;
        }
    }
}
