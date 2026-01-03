#include "./layer.h"
#include "./loss.h"
#include "./optimizer.h"
#include <vector>

/**
 * Class that configures the Neural Network
 */
class Network {
    private:
        std::vector<Layer *> layers;
        Loss *loss;

    public:
        // constructor
        Network() : loss(NULL) {}

        // destructor
        ~Network() {
            for (int i = 0; i < layers.size(); i++) {
                delete layers[i];
            }
            if (loss) {
                delete loss;
            }
        }

        void add_layer(Layer* layer) { layers.push_back(layer); }
        void add_loss(Loss* loss_in) { loss = loss_in; };

        void forward(const Matrix &input);
        void backward(const Matrix &input, const Matrix &target);

        void update(Optimizer *opt);

        Matrix &output() { layers.back()->output(); };
};
