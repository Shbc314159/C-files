#include "Connection.h"
#include "GlobalVars.h"

Connection::Connection(int input_neuron, int output_neuron, int innovation_number)
    : input_neuron(input_neuron), output_neuron(output_neuron), innovation_number(innovation_number)
{
    weight = globalvars::generateRandomNumber(-2.0, 2.0);
    active = true; 
}