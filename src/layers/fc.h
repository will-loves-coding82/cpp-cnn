#include "../layer.h"
#include <Eigen/Dense>

class FC : public Layer {
    protected:
        Matrix weight;
        Vector bias;

        Matrix grad_weight;
        Vector grad_bias;

        // Input and output dimensions
        int d_in, d_out;

        void init();

    public:

    


};