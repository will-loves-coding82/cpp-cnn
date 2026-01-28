#ifndef CROSS_ENTROPY_H_
#define CROSS_ENTROPY_H_

#include "../loss.h"

class CrossEntropy : public Loss
{
public:
    void evaluate(const Matrix &pred, const Matrix &target);
};

#endif