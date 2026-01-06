#ifndef SRC_LAYER_H_
#define SRC_LAYER_H_

#include <Eigen/Core>
#include <vector>
#include "./utils.h"
#include "./optimizer.h"

// https://github.com/iamhankai/mini-dnn-cpp/blob/master/src/layer.h

/**
 * Every layer instance is responsible for implementing a custom
 * forward and backward pass logic. 
 */
class Layer
{
protected:
    Matrix top;         // layer output
    Matrix grad_bottom; // gradient w.r.t input

public:
    virtual ~Layer() {}

    virtual void forward(const Matrix &bottom) = 0;
    
    // Computes the grad_bottom w.r.t to this layer's input denoted by 'bottom'
    // grad_top denotes the gradient from the layer above. This is part of the 
    // back propogation derivative chain
    virtual void backward(const Matrix &bottom, const Matrix &grad_top) = 0;

    // Updates this layer's grad
    virtual void update(Optimizer &opt) {}

    // Returns a layer's output matrix
    virtual const Matrix &output() { return top; }
    virtual const Matrix &back_gradient() { return grad_bottom; }
    virtual int output_dim() { return -1; }

    virtual std::vector<float> get_parameters() const
    {
        return std::vector<float>();
    }
    
    virtual std::vector<float> get_derivatives() const
    {
        return std::vector<float>();
    }
    virtual void set_parameters(const std::vector<float> &param) {}
};

#endif // SRC_LAYER_H_