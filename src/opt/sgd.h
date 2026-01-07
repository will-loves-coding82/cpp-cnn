#include "../optimizer.h"
#include "../utils.h"

class SGD : Optimizer {
    public:
        SGD(float lr) : Optimizer(lr){};
        void update(Vector::AlignedMapType &param, Vector::ConstAlignedMapType &grad);
};