#include "Neural_Network.h"

#include <vector>
#include <algorithm>

class Species {
public:
    double highest_fitness = 0.0;
    int last_gen_improvement = 0;
    int current_generation = 1;
    Neural_Network* template_network;
    std::vector<Neural_Network*> current_members;
    double c1;
    double c2;
    double c3;
    double compatibility_threshold;
    double total_fitness = 0.0;

 
    Species(Neural_Network* template_network, double c1, double c2, double c3, double compatibility_threshold) {
        this->template_network = template_network;
        current_members.push_back(template_network);
        this->c1 = c1;
        this->c2 = c2;
        this->c3 = c3;
        this->compatibility_threshold = compatibility_threshold;
    }

    bool check_network(Neural_Network* new_network) {
        double num_excess = 0;
        double num_disjoint = 0;
        double weight_difference_matching = 0;
        double genes_in_larger = 0;
        new_network->print();
        template_network->print();

        if (new_network->genome_connections.size() > template_network->genome_connections.size()) {
            double genes_in_larger = new_network->genome_connections.size();
        } else {
            double genes_in_larger = template_network->genome_connections.size();
        }

        int newIndex = 0;
        int originalIndex = 0;

        while (newIndex < new_network->genome_connections.size() && originalIndex < template_network->genome_connections.size())
        {
            const Connection *newConn = new_network->genome_connections[newIndex];
            const Connection *originalConn = template_network->genome_connections[originalIndex];

            if (newConn->innovation_number == originalConn->innovation_number)
            {
                weight_difference_matching += fabs(newConn->weight - originalConn->weight);
                newIndex++;
                originalIndex++;
            }
            else if (newConn->innovation_number < originalConn->innovation_number)
            {
                num_disjoint++;
                newIndex++;
            }
            else
            { // newConn->innovation_number > originalConn->innovation_number
                num_excess++;
                originalIndex++;
            }
        }

        // Count any remaining excess genes in the new genome
        num_excess += (new_network->genome_connections.size() - newIndex);

        double difference = (c1*num_excess/genes_in_larger) + (c2*num_disjoint/genes_in_larger) + (c3*weight_difference_matching);

        if (difference < compatibility_threshold) {
            return true;
        } else {
            return false;
        }
    }

    void add_network(Neural_Network* new_network) {
        current_members.push_back(new_network);
    }

    void sort_species() {
        std::sort(current_members.begin(), current_members.end(), [](const Neural_Network* obj1, const Neural_Network* obj2)
                  { return obj1->fitness > obj2->fitness; });
    }

    void remove_worst(int number_of_members) {
        sort_species();
        current_members.erase(current_members.end() - number_of_members, current_members.end());
    }

    void increment_generation() {
        sort_species();
        for (auto& member : current_members) {
            if (member->fitness > highest_fitness) {
                highest_fitness = member->fitness;
                last_gen_improvement = current_generation;
            }
        }

        template_network = current_members[0];
        current_members.clear();
        current_generation++;
    }

    bool kill_species() {
        if (last_gen_improvement < current_generation - 15) {
            return true;
        } else {
            return false;
        }
    }

};
