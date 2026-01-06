#include "./softmax_loss.h"
#include <Eigen/Dense>

/**
 * Updates the new cross entropy loss value after a forward pass. The loss's 
 * grad_bottom is also calculated for back propogation.
 */ 
void SoftMaxLoss::evaluate (const Matrix& pred, const Matrix& target) {
    int classes = target.cols();

    // Softmax Loss = \sum{-yi * log(pi)} / n
    // (Y * log(P)).sum() / n
    loss = target.array().cwiseProduct(pred.array().log()).sum() / classes;

    // The grad_bottom for the loss function is used to
    // compute the back propogation for the lower layers
    // dL/dP = -Y / P / n
    grad_bottom = target.array().cwiseQuotient(pred.array()) / classes;
}