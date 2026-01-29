#include "./cross_entropy.h"
#include <Eigen/Dense>
#include <iostream>
/**
 * Updates the average loss and accuracy after a complete forward pass. 
 * The loss's grad_bottom is also calculated for back propogation.
 */ 
void CrossEntropy::evaluate (const Matrix& pred, const Matrix& target) {
    int batch_size = pred.cols();

    // Average Cross Entropy Loss = \sum{-yi * log(pi)} / n
    // (Y * log(P)).sum() / n
    loss = -(target.array().cwiseProduct((pred.array() + 1e-8).log())).sum();
    loss /= batch_size;

    // Compute the average accuracy for the batch

    int numCorrect = 0;
    for (int i = 0; i < batch_size; i++)
    {
        Eigen::MatrixXf::Index max_index;
        Eigen::MatrixXf::Index true_index;
        pred.col(i).maxCoeff(&max_index);
        target.col(i).maxCoeff(&true_index);

        // DEBUG: Print first 5 samples
        if (i < 5)
        {
            std::cout << "Sample " << i << ": predicted=" << max_index
                      << ", true=" << true_index
                      << ", confidence=" << pred(max_index, i) << std::endl;
        }

        if (max_index == true_index)
        {
            numCorrect++;
        }
    }

    accuracy = float(numCorrect) / batch_size * 100.0f;

    // The grad_bottom for the loss function is used to
    // compute the back propogation for the lower layers
    // dL/dP = -Y / P / n
    grad_bottom = -target.array().cwiseQuotient(pred.array() + 1e-8) / batch_size;
}