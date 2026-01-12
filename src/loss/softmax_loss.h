#ifndef SOFTMAX_LOSS_H_
#define SOFTMAX_LOSS_H_

#include "../loss.h"

class SoftMaxLoss : public Loss
{
public:
    void evaluate(const Matrix &pred, const Matrix &target);
};

#endif