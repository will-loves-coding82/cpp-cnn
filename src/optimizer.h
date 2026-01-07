#include <Eigen/Core>
#include <Eigen/Dense>

class Optimizer {
    public:
        float learning_rate;

        Optimizer(float lr) : learning_rate(lr){};

        /**
         * @brief Updates a layer's parameters.
         * @param param is a Vector::AlignedMapType which allows Eigen to treat it as a raw memory block.
         * @param grad is a Vector::ConstAlignedMapType which is similar as the above but is also immutable.
         * 
         * @return void
         *  
         * @details
         * The Aligned keyword indicates that the memory pointer is aligned allowing Eigen to use
         * faster SIMD instructions
         */
        void update(Vector::AlignedMapType& param, Vector::ConstAlignedMapType& grad);
};