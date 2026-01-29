#ifndef LAYER_H_
#define LAYER_H_

#include "Eigen/Core"
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
    Matrix top;         // every layer has an output
    Matrix grad_bottom; // gradient w.r.t input

public:

    // Deconstructor that applies for all derived class types
    virtual ~Layer() {};

    // The = 0 syntax makes these pure virtual functions. 
    // They have no implementation here—derived classes must override them. 
    // This makes Layer an abstract base class that can't be instantiated directly.
    virtual void forward(const Matrix &bottom) = 0;

    // Computes the grad_bottom w.r.t to this layer's input denoted by 'bottom'
    // grad_top denotes the gradient from the layer above. This is part of the 
    // back propogation derivative chain
    virtual void backward(const Matrix &bottom, const Matrix &grad_top) = 0;

    // The following are virtual function implementations. Note that if a function
    // has an empty body it means it's not always required. For example, a ReLU
    // layer has no weights to update.
    
    virtual void update(Optimizer &opt) {};
    virtual Matrix get_grad_weight() {};
    virtual const Matrix &output() { return top; };
    virtual const Matrix &back_gradient() { return grad_bottom; };
    virtual int output_dim() { return -1; };
};

#endif // LAYER_H_