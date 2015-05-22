#include <cstddef>
#include <iostream>

#include "neuronet/neural_connection.hpp"
#include "neuronet/neural_layer.hpp"
#include "neuronet/neural_net.hpp"
#include "neuronet/neuron.hpp"

#include "utility/training_data.hpp"

int main() {
	auto data = utility::TrainingData{"../test/test_01.data"};
	auto net  = nn::NeuralNet{data.getTopology()};
	std::cout << "Input Topology = ";
	for (auto&& elem : data.getTopology()) {
		std::cout << elem << ' ';
	}   std::cout << '\n';
	for (auto&& pass : data) {
		std::cout << "InputValues = ";
		for (auto&& elem : pass.getInputValues()) {
			std::cout << elem << ' ';
		}   std::cout << '\n';
		std::cout << "ExpectedValues = ";
		for (auto&& elem : pass.getExpectedValues()) {
			std::cout << elem << ' ';
		}   std::cout << '\n';
		net.feedForward(pass.getInputValues());
		net.backPropagation(pass.getExpectedValues());
	}

	return 0;
}
