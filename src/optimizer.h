#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include "./utils.h"

class Optimizer {
    public:
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
        virtual void update(Vector::AlignedMapType& param, Vector::ConstAlignedMapType& grad) {};
};

#endif