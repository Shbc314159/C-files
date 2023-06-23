#include "GlobalVars.h"
#include <random>
#include <ctime>
 
namespace globalvars
{
    int next_innovation_number = 1;
    int next_id = 4;
    std::vector<Neuron *> neurons;
    std::vector<Connection *> connections;
    std::mt19937 rng;
 
    void initializeRandomGenerator()
    {
        std::random_device rd;
        rng.seed(static_cast<unsigned int>(std::time(nullptr)));
    }

    double generateRandomNumber(double lowerBound, double upperBound)
    {
        std::uniform_real_distribution<double> dist(lowerBound, upperBound + std::numeric_limits<double>::epsilon());
        return dist(rng);
    }
}