#include "./cross_entropy.h"
#include <Eigen/Dense>
#include <iostream>
/**
 * Updates the new cross entropy loss value after a forward pass. The loss's 
 * grad_bottom is also calculated for back propogation.
 */ 
void CrossEntropy::evaluate (const Matrix& pred, const Matrix& target) {
    std::cout << "CrossEntropyLoss" << std::endl;
    std::cout << "pred.cols(): " << pred.cols() << std::endl;
    std::cout << "pred.rows(): " << pred.rows() << std::endl;

    std::cout << "target.cols(): " << target.cols() << std::endl;
    std::cout << "target.rows(): " << target.rows() << std::endl;

    int classes = pred.cols();

    // Cross Entropy Loss = \sum{-yi * log(pi)} / n
    // (Y * log(P)).sum() / n
    loss = -target.array().cwiseProduct((pred.array() + 1e-8).log()).sum();
    loss /= classes;
    std::cout << "calculated loss: " << loss << std::endl;

    // The grad_bottom for the loss function is used to
    // compute the back propogation for the lower layers
    // dL/dP = -Y / P / n
    grad_bottom = -target.array().cwiseQuotient(pred.array()) / classes;
    // std::cout << "calculated loss grad_bottom: " << grad_bottom <<  std::endl;
}