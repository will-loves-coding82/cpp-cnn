#ifndef SRC_CROSS_ENTROPY_LOSS_H_
#define SRC_CROSS_ENTROPY_LOSS_H_

#include "../loss.h"

class CrossEntropyLoss : public Loss
{
public:
    void evaluate(const Matrix &pred, const Matrix &target);
};

#endif