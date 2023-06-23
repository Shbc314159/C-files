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

double expected_output(std::vector<double> inputs) {
    if ((inputs[0] == 0.0 & inputs[1] == 0.0) | (inputs[1] == 1.0 & inputs[1] == 1.0))
    {
        return 0.0;
    } else {
        return 1.0;
    }
}

int main() {
    Genetic_Algorithm ga = Genetic_Algorithm(10, { 0.1, 0.03, 0.25, 0.72, 0.08 }, 2, 1, 1.0, 1.0, 0.4, 3.0);
    ga.initialise_population();
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;
    std::vector<double> differences;

    for (int l = 0; l != 100; l++) {
        inputs.clear();
        outputs.clear();
        differences.clear();
        std::cout << "Generation: " << ga.current_generation << std::endl;

        for (int k = 0; k != 100; k++) {
            for (int i = 0; i != 10; i++) {
                std::vector<double> input = { static_cast<double>(static_cast<int>(globalvars::generateRandomNumber(0, 2))), static_cast<double>(static_cast<int>(globalvars::generateRandomNumber(0, 2))) };
                inputs.push_back(input);
            }

            outputs = ga.run_population(inputs);

            for (int j = 0; j != outputs.size(); j++) {
                std::vector<double> inputs_pair = inputs[j];
                double output = outputs[j][0];
                double best_output = expected_output(inputs_pair);
                double difference = std::abs(output - best_output);
                if (differences.size() != outputs.size()) {
                    differences.push_back(-difference);
                } else {
                    differences[j] -= difference;
                }
            }
        }

        ga.set_fitnesses(differences);

        Neural_Network* best = ga.get_best_network();
        std::vector<double> best_inputs = {0.0, 0.0};
        std::cout << best->run(best_inputs)[0] << std::endl;
        best_inputs = {1.0, 0.0};
        std::cout << best->run(best_inputs)[0] << std::endl;
        best_inputs = {0.0, 1.0};
        std::cout << best->run(best_inputs)[0] << std::endl;
        best_inputs = {1.0, 1.0};
        std::cout << best->run(best_inputs)[0] << std::endl;


        ga.create_next_generation();
    }

    return 0;
}