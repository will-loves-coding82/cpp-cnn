#ifndef SGD_H_
#define SGD_H_

#include "../optimizer.h"
#include "../utils.h"

class SGD : public Optimizer {
    protected:
        float learning_rate;
    public:
        explicit SGD(float lr) : learning_rate(lr) { };
        void update(Vector::AlignedMapType &param, Vector::ConstAlignedMapType &grad);
};

#endif