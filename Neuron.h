#ifndef NEURON_H
#define NEURON_H

#include <cfloat>
#include <cmath>
#include <iostream>

class Neuron {
    public:
        int id; 
        int input_neuron;
        int output_neuron; 
        double sum = 0;
        double value = DBL_MAX;
        
        Neuron(int id, int input_neuron, int output_neuron) {
            this->id = id;
            this->input_neuron = input_neuron;
            this->output_neuron = output_neuron;
        }
    
        void activate() {
            value = sigmoid(sum);
        }

        double sigmoid(double x) {
            double z = clip(x, -20, 20);
            return 1 / (1 + exp(-z));
        }

        double clip(double value, int min_value, int max_value) {
            if (value < min_value) {
                return min_value;
            } else if (value > max_value) {
                return max_value;
            } else {
                return value;
            }
        }

        void print() {
            std::cout << "Neuron " << id << ": " << input_neuron << " -> " << output_neuron << " " << sum << value << std::endl;
        }
};

#endif // NEURON_H