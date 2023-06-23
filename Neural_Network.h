#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Neuron.h" 
#include "Connection.h"
#include "GlobalVars.h"

#include <vector>
#include <utility>
#include <map>  
#include <set>
#include <limits>
#include <unordered_map>
#include <iostream>
#include <random>
#include <functional>

class Neural_Network {
public:
    std::vector<Neuron*> genome_neurons;
    std::vector<Connection*> genome_connections;
    std::vector<Neuron*> input_neurons;
    std::vector<Neuron*> output_neurons;
    std::vector<Neuron*> hidden_neurons;
    double fitness = 0;
    int num_inputs;
    int num_outputs;

    Neural_Network(int num_inputs, int num_outputs) {
        this->num_inputs = num_inputs;
        this->num_outputs = num_outputs;
        setup_layers();
    }

    void set_fitness() {
        fitness -= (genome_connections.size() * genome_neurons.size())/100;
    }

    Neural_Network* __copy__() {
        Neural_Network othernetwork = Neural_Network(this->num_inputs, this->num_outputs);
        othernetwork = *this;
        Neural_Network* clone = this->crossover(othernetwork);
        return clone;
    }

    void setup_layers() {
        for (int i = 0; i != num_inputs; i++) {
            Neuron* neuron = new Neuron(i, -1, -1);
            genome_neurons.push_back(neuron);
            input_neurons.push_back(neuron);
            add_to_global_list(neuron);
        }

        for (int i = num_inputs+1; i != num_inputs+num_outputs+1; i++) {
            Neuron* neuron = new Neuron(i, -1, -1);
            genome_neurons.push_back(neuron);
            output_neurons.push_back(neuron);
            add_to_global_list(neuron);
        }

        Neuron* neuron = new Neuron(num_inputs, -1, -1);
        genome_neurons.push_back(neuron);
        input_neurons.push_back(neuron);
        add_to_global_list(neuron);
    }

    Connection* create_connection(int input_id, int output_id, double weight=0.0, bool active=false, bool pass_weight=false, bool pass_activation=false) {
        bool new_connection = true;
        bool connection_in_network = false;
        int innovation_number;

        for (auto& connection : globalvars::connections) {
            if (connection->input_neuron == input_id & connection->output_neuron == output_id)
            {
                new_connection = false;
                innovation_number = connection->innovation_number;
            }
        }

        if (new_connection == true) {
            innovation_number = globalvars::next_innovation_number;
            globalvars::next_innovation_number++;
        }

        Connection* connection = new Connection(input_id, output_id, innovation_number);

        if (pass_weight == true) {
            connection->weight = weight;
        }

        if (pass_activation == true) {
            connection->active = active;
        }

        for (auto& otherconnection: genome_connections) {
            if (otherconnection->innovation_number == connection->innovation_number)
            {
                connection_in_network = true;
                otherconnection->weight += connection->weight;
            }
        }

        if (connection_in_network == false) {
            genome_connections.push_back(connection);
        }

        if (new_connection == true) {
            globalvars::connections.push_back(connection);
        }

        return connection;
    }

    std::pair<Connection*, Connection*> create_node(Connection* connection) {
        int input_id = connection->input_neuron;
        int output_id = connection->output_neuron;
        bool new_neuron = true;
        bool neuron_in_network = false;
        int neuron_id;

        for (auto& neuron : globalvars::neurons) {
            if (neuron->input_neuron == input_id & neuron->output_neuron == output_id)
            {
                new_neuron = false;
                neuron_id = neuron->id;
            }
        }

        if (new_neuron == true) {
            neuron_id = globalvars::next_id;
            globalvars::next_id++;
        }

        connection->active = false;
        Neuron* neuron = new Neuron(neuron_id, input_id, output_id);

        if (new_neuron) {
            globalvars::neurons.push_back(neuron);
        }

        for (auto& otherneuron : genome_neurons) {
            if (otherneuron->id == neuron_id)
            {
                neuron_in_network = true;
            }
        }

        if (neuron_in_network == false) {
            genome_neurons.push_back(neuron);
        }

        Connection* connection1 = create_connection(input_id, neuron->id);
        Connection* connection2 = create_connection(neuron->id, output_id);
        connection1->weight = 1;
        connection2->weight = connection->weight;

        return std::make_pair(connection1, connection2);
    }

    std::vector<double> run(std::vector<double> inputs) {
        std::map<int, Neuron*> neuron_map;
        reset();

        for (auto& neuron : genome_neurons) {
            neuron_map.emplace(neuron->id, neuron);
        }

        std::vector<double> outputs;
        std::vector<Connection*> active_connections;
        std::set<int> visited;
        std::vector<double> examples = { 1, 1, 1, 1};

        for (int i=0; i!=num_inputs; i++) {
            input_neurons[i]->value = inputs[i];
        }

        input_neurons[num_inputs]->value = 1;

        for (auto& connection : genome_connections) {
            if (connection->active == true) {
                active_connections.push_back(connection);
            }
        }

        for (auto& neuron : output_neurons) {
            outputs.push_back(get_neuron_value(neuron->id, active_connections, neuron_map, visited));
            visited.clear();
        }

        return outputs;
    }

    double get_neuron_value(int neuron_id, std::vector<Connection*> active_connections, std::map<int, Neuron*> neuron_map, std::set<int> visited) {

        Neuron* neuron = neuron_map.at(neuron_id);

        if (!std::isnan(neuron->value)) {
            return neuron->value;
        }

        if (visited.count(neuron_id) > 0) {
            return 1; //this could have an impact on results as it determines the result of any circular synapse structure
        } else {
            visited.insert(neuron_id);
        }

        for (auto& connection : active_connections) {

            if (connection->output_neuron == neuron->id) {
                Neuron* input_neuron = neuron_map.at(connection->input_neuron);

                if (std::isnan(input_neuron->value)) {
                    neuron->sum += get_neuron_value(connection->input_neuron, active_connections, neuron_map, visited) * connection->weight;
                }
                else {
                    neuron->sum += input_neuron->value * connection->weight;
                }
            }
        }

        neuron->activate();

        return neuron->value;
    }

    void reset() {
        for (auto& neuron : genome_neurons) {
            neuron->sum = 0;
            neuron->value = std::numeric_limits<double>::quiet_NaN();
        }
    }

    Neural_Network* crossover(Neural_Network other_network) {
        Neural_Network* offspring = new Neural_Network(num_inputs, num_outputs);
        std::vector<Connection *> matching_genes = match_genes(other_network);

        if (fitness > other_network.fitness) {
            for (auto& connection : genome_connections) {
                bool connection_in_network = false;

                for (auto& matching_connection : matching_genes) {
                    if (matching_connection->innovation_number == connection->innovation_number) {
                        connection_in_network = true;
                    }
                }

                if (connection_in_network == false) {
                    matching_genes.push_back(connection);
                }
            }
        } else {
            for (auto &connection : other_network.genome_connections) {
                bool connection_in_network = false;

                for (auto &matching_connection : matching_genes) {
                    if (matching_connection->innovation_number == connection->innovation_number) {
                        connection_in_network = true;
                    }
                }

                if (connection_in_network == false) {
                    matching_genes.push_back(connection);
                }
            }
        }

        for (auto &connection : matching_genes) {
            Connection* offspring_connection = offspring->create_connection(connection->input_neuron, connection->output_neuron, connection->weight, connection->active, true, true);

            if (has_node(*offspring, connection->input_neuron) == false) {
                int id = connection->input_neuron;
                int input_id, output_id = find_input_output_neuron(id);

                Neuron* neuron = new Neuron(id, input_id, output_id);
                offspring->genome_neurons.push_back(neuron);

                if ((id < num_inputs) || (id == num_inputs + num_outputs)) {
                    offspring->input_neurons.push_back(neuron);
                } else if ((id < num_inputs + num_outputs) && (id >= num_inputs)) {
                    offspring->output_neurons.push_back(neuron);
                } else { 
                    offspring->hidden_neurons.push_back(neuron);
                }
            }  

            if (has_node(*offspring, connection->output_neuron) == false) {
                int id = connection->output_neuron;
                int input_id, output_id = find_input_output_neuron(id);

                Neuron *neuron = new Neuron(id, input_id, output_id);
                offspring->genome_neurons.push_back(neuron);

                if ((id < num_inputs) || (id == num_inputs + num_outputs)) {
                    offspring->input_neurons.push_back(neuron);
                } else if ((id < num_inputs + num_outputs) && (id >= num_inputs)) {
                    offspring->output_neurons.push_back(neuron);
                } else {
                    offspring->hidden_neurons.push_back(neuron);
                }
            }
        } 

        return offspring;
    }

    std::vector<Connection*> match_genes(Neural_Network other_network) {
        std::vector<Connection*> matching_genes;
        
        for (auto& connection : genome_connections) {
            for (auto& otherconnection : other_network.genome_connections) {
                if (connection->innovation_number == otherconnection->innovation_number) {
                    if (fitness > other_network.fitness) {
                        matching_genes.push_back(connection); 
                    } else { 
                        matching_genes.push_back(otherconnection);
                    }
                    break;
                }
            }
        } 
        return matching_genes;
    }

    bool has_node(Neural_Network network, int node_id) {
        for (auto& neuron : network.genome_neurons) {
            if (neuron->id == node_id) {
                return true;
            }
        }

        return false;
    }

    bool find_input_output_neuron(int id) {
        for (auto& neuron : globalvars::neurons) {
            if (neuron->id == id) {
                return neuron->input_neuron, neuron->output_neuron;
            }
        }
        std::cout << std::endl << "Error: Neuron not found" << std::endl;
    }

    void add_to_global_list(Neuron* neuron) {
        for (auto& neuron_global : globalvars::neurons) {
            if (neuron_global->id == neuron->id) {
                return;
            }
        }
        globalvars::neurons.push_back(neuron);
    }  

    void mutate_connection() {
        Neuron* input_neuron;
        Neuron* output_neuron;

        while (true) {
            int non_output_randomIndex = static_cast<int>(globalvars::generateRandomNumber(0, genome_neurons.size()));
            input_neuron = genome_neurons[non_output_randomIndex];

            if (input_neuron->id <= num_inputs | input_neuron->id > num_inputs + num_outputs) {
                break;
            }
        }

        while (true) {
            int output_randomIndex = static_cast<int>(globalvars::generateRandomNumber(0, genome_neurons.size()));
            output_neuron = genome_neurons[output_randomIndex];

            if (output_neuron->id > num_inputs & output_neuron->id <= num_inputs + num_outputs) {
                break;
            }
        }

        if (output_neuron->id == input_neuron->id) {
            std::cout << std::endl << "Error: Input and output neurons cannot be the same" << std::endl;
        }

        Connection* connection = create_connection(input_neuron->id, output_neuron->id);
    }  

    void mutate_neuron() {
        Connection* conn;
        int index = static_cast<int>(globalvars::generateRandomNumber(0, genome_connections.size()));
        conn = genome_connections[index];
        create_node(conn);
    }

    void mutate_enable_disable() {
        Connection *conn;
        int index = static_cast<int>(globalvars::generateRandomNumber(0, genome_connections.size()));
        conn = conn = genome_connections[index];
        if (conn->active == true) {
            conn->active = false;
        }
        else {
            conn->active = true;
        }
    }

    void mutate_weight_shift() {
        Connection *conn;
        int index = static_cast<int>(globalvars::generateRandomNumber(0, genome_connections.size()));
        conn = conn = genome_connections[index];
        conn->weight *= globalvars::generateRandomNumber(-2, 2);
    }

    void mutate_weight_random() {
        Connection *conn;
        int index = static_cast<int>(globalvars::generateRandomNumber(0, genome_connections.size()));
        conn = conn = genome_connections[index];
        conn->weight = globalvars::generateRandomNumber(-2, 2);
    }

    void mutate(std::vector<double> probabilities) {
        if (probabilities[0] > globalvars::generateRandomNumber(0, 1)) {
            mutate_connection();
        }

        if (probabilities[1] > globalvars::generateRandomNumber(0, 1) & genome_connections.size() > 0) {
            mutate_neuron();
        }

        if (probabilities[2] > globalvars::generateRandomNumber(0, 1) & genome_connections.size() > 0) {
            mutate_enable_disable();
        }

        if (probabilities[3] > globalvars::generateRandomNumber(0, 1) & genome_connections.size() > 0) {
            mutate_weight_shift();
        }

        if (probabilities[4] > globalvars::generateRandomNumber(0, 1) & genome_connections.size() > 0) {
            mutate_weight_random();
        }
    }

    void print() {
        std::cout << std::endl << "Neural Network:" << std::endl << std::endl;
        
        std::cout << "Genome: " << std::endl;
        for (auto& neuron : genome_neurons) {
            neuron->print();
        }
        for (auto& connection : genome_connections) {
            connection->print();
        }
        std::cout << "Input neurons:" << std::endl;
        for (auto& neuron : input_neurons) {
            neuron->print();
        }
        std::cout << "Hidden neurons:" << std::endl;
        for (auto &neuron : hidden_neurons)
        {
            neuron->print();
        }
        std::cout << "Output neurons:" << std::endl;
        for (auto &neuron : output_neurons)
        {
            neuron->print();
        }
        std::cout << "Num inputs: " << num_inputs << " Num outputs: " << num_outputs << " Fitness: " << fitness << std::endl;
    }
};

#endif