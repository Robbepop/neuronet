#include <cstddef>
#include <iostream>

#include "nn/neural_connection.hpp"
#include "nn/neural_layer.hpp"
#include "nn/neural_net.hpp"
#include "nn/neuron.hpp"

int main() {
	std::cout << "Hello, neural network!\n";

	auto topology = std::vector<size_t>{2,4,1};
	auto net = nn::NeuralNet{topology};

	return 0;
}
