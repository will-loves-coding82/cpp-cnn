#ifndef LOSS_H_
#define LOSS_H_

#include "./utils.h"

class Loss {
    protected:
        float loss;

        // Gradient w.r.t input. Used for back propogation starting from 
        // the last output layer
        Matrix grad_bottom;

    public:
        // Unlike other member functions where you might override them explicitly, 
        // destructors have fixed names (~ClassName). The compiler won't automatically 
        // chain them polymorphically without the virtual keyword.
        virtual ~Loss() {}
        virtual void evaluate(const Matrix &pred, const Matrix &target) = 0;
        virtual float output() { return loss; }
        virtual const Matrix &back_gradient() { return grad_bottom; }
};

#endif