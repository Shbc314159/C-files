#include "Neuron.h"
#include "Connection.h"
#include "GlobalVars.h"
#include "Neural_Network.h"
#include "Species.h"

class Genetic_Algorithm {
public:
    int pop_size;
    int selection_size;
    Neural_Network* best_network;
    int current_generation = 1;
    std::vector<Neural_Network*> population;
    std::vector<double> mutation_probs;
    int num_inputs;
    int num_outputs;
    std::vector<Species> species_list;
    double c1, c2, c3, compatibility_threshold;

    Genetic_Algorithm (int pop_size, std::vector<double> mutation_probs, int num_inputs, int num_outputs, double c1, double c2, double c3, double compatibility_threshold) {
        this->num_inputs = num_inputs;
        this->num_outputs = num_outputs;
        this->pop_size = pop_size;
        this->mutation_probs = mutation_probs;
        globalvars::initializeRandomGenerator();
        globalvars::next_id = num_inputs + num_outputs + 1;
        initialise_population();
        selection_size = pop_size * 0.1;
        this->c1 = c1;
        this->c2 = c2;
        this->c3 = c3;
        this->compatibility_threshold = compatibility_threshold;
    }

    void initialise_population() {
        for (int i = 0; i != pop_size; i++) {
            Neural_Network* network = new Neural_Network(num_inputs, num_outputs);
            network->mutate_connection();
            population.push_back(network);   
        }
    }

    void create_next_generation() {
        std::vector<Neural_Network*> next_gen = speciation();
        population.clear();

        for (auto& network : next_gen) {
            population.push_back(network);
        }

        current_generation++;
    }

    std::vector<Neural_Network*> speciation() {
        int num_needed = pop_size;
        networks_fitness_with_complexity();
        std::vector<Neural_Network*> next_gen_networks;

        for (Neural_Network* network : population) {
            for (Species species : species_list) {
                if (species.check_network(network) == true) {
                    std::cout << "adding network" << std::endl;
                    species.add_network(network);
                    break;
                }
            }

            Species new_species = Species(network, c1, c2, c3, compatibility_threshold);
            species_list.push_back(new_species);
        }

        double total_fitness = 0;

        for (Species species : species_list) {
            species.total_fitness = 0.0;
            for (Neural_Network* network : species.current_members) {
                network->fitness /= species.current_members.size();
                total_fitness += network->fitness;
                species.total_fitness += network->fitness;
            }
        }

        for (int i = 0; i != species_list.size(); i++) {
            Species species = species_list[i];
            int num_to_keep = (species.total_fitness / total_fitness) * num_needed;
            int num_to_remove = species.current_members.size() - num_to_keep;
            int num_offspring = num_to_keep;
            species.sort_species();

            if (species.current_members.size() >= 5) {
                next_gen_networks.push_back(species.current_members[0]->__copy__());
                num_to_keep -= 1;
                num_offspring -= 1;
            }

            species.remove_worst(num_to_remove);

            for (int i = 0; i!= num_offspring; i++) {
                int index1 = globalvars::generateRandomNumber(0, species.current_members.size() - 1);
                int index2 = globalvars::generateRandomNumber(0, species.current_members.size() - 1);
                Neural_Network parent1 = *species.current_members[index1];
                Neural_Network parent2 = *species.current_members[index2];
                Neural_Network* offspring = parent1.crossover(parent2);
                next_gen_networks.push_back(offspring);
            }

            species.increment_generation();
            if (species.kill_species() == true) {
                species_list.erase(species_list.begin() + i);
            }
        }

        return next_gen_networks;
    }

    std::vector<std::vector<double>> run_population(std::vector<std::vector<double>> inputs) {
        std::vector<std::vector<double>> outputs;
        for (int i = 0; i!= pop_size; i++) {
            Neural_Network* network = population[i];
            outputs.push_back(network->run(inputs[i]));
        }

        return outputs;
    }

    Neural_Network* get_best_network() {
        double highest = -10000000000;
        Neural_Network* best_network;
        for (auto& network : population) {
            if ((network->fitness -= (network->genome_connections.size() * network->genome_neurons.size() / 100)) > highest) {
                highest = network->fitness;
                best_network = network;
            }
        }

        return best_network;
    }

    void set_fitnesses(std::vector<double> fitnesses) {
        for (int i = 0; i!= pop_size; i++) {
            Neural_Network* network = population[i];
            network->fitness = fitnesses[i];
        }
    }

    void networks_fitness_with_complexity() {
        for (Neural_Network* network : population) {
            network->set_fitness();
        }
    }

};