#ifndef CONNECTION_H
#define CONNECTION_H

#include <iostream>
 
class Connection {
public:
    float weight;
    bool active;
    int input_neuron;
    int output_neuron;
    int innovation_number;

    Connection(int input_neuron, int output_neuron, int innovation_number);

    void print() {
        std::cout << "Connection " << innovation_number << ": " << input_neuron << ", " << output_neuron << ", " << weight << ", " << active << std::endl;
    }
};

#endif // CONNECTION_H