#ifndef GLOBALVARS_H
#define GLOBALVARS_H

#include "Neuron.h"
#include "Connection.h"

#include <vector>
#include <random>

namespace globalvars
{
    extern int next_innovation_number;
    extern int next_id;
    extern std::vector<Neuron *> neurons;
    extern std::vector<Connection *> connections;

    extern std::mt19937 rng;

    void initializeRandomGenerator();
    double generateRandomNumber(double lowerBound, double upperBound);
}

#endif
