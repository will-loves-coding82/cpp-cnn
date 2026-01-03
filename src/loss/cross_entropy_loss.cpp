#include "./cross_entropy_loss.h"
#include <Eigen/Dense>

void CrossEntropyLoss::evaluate (const Matrix& pred, const Matrix& target) {
    int classes = target.cols();

    // Softmax Loss = \sum{-yi * log(pi)} / n
    // (Y * log(P)).sum() / n
    loss = target.array().cwiseProduct(pred.array().log()).sum() / classes;

    // The grad_bottom for the loss function is used to
    // compute the back propogation through the lower layers
    // dL/dP = -Y / P / n
    grad_bottom = target.array().cwiseQuotient(pred.array()) / classes;
}