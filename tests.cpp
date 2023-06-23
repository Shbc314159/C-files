#include "Neuron.h"
#include "Connection.h"
#include "GlobalVars.h"
#include "Neural_Network.h"
#include "Genetic_Algorithm.h"

#include <vector>
#include <utility>
#include <map>
#include <set>
#include <limits>
#include <unordered_map>
#include <iostream>
#include <random>
#include <functional>
#include <cmath>

int main() {
    std::vector<double> probabilities = { 0.5, 0.5, 0.5, 0.5, 0.5 };
    Neural_Network network = Neural_Network(2, 2);
    network.mutate(probabilities);

    Species species = Species(&network, 1.0, 1.0, 0.4, 3.0);
    
    for (int i = 0; i < 100000; i++) {
        Neural_Network* network = new Neural_Network(2, 2);
        network->mutate(probabilities);

        if (species.check_network(network)) {
            species.current_members.push_back(network);
        }
    }

    std::cout << species.current_members.size() << std::endl;


    return 0;
}