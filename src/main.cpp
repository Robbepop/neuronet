#include <cstddef>
#include <iostream>
#include <cassert>
#include <chrono>

#include <vector>

#include "neuronet/neural_connection.hpp"
#include "neuronet/neural_layer.hpp"
#include "neuronet/neural_net.hpp"
#include "neuronet/neuron.hpp"

#include "utility/training_data.hpp"
#include "utility/print_vector.hpp"

int main(int argc, const char ** argv) {
	if (argc < 2) throw std::runtime_error{"too few parameters passed to program!"};
	auto data = utility::TrainingData{argv[1]};
	auto net  = nn::NeuralNet{data.getTopology()};
	std::cout << "Input Topology = " << data.getTopology() << '\n' << '\n';

	const auto start = std::chrono::steady_clock::now();

	for (auto&& pass : data) {
		//std::cout << "InputValues    = " << pass.getInputValues() << '\n';
		//std::cout << "ExpectedValues = " << pass.getExpectedValues() << '\n';
		net.feedForward(pass.getInputValues());
		net.backPropagation(pass.getExpectedValues());
		//auto results = net.results();
		//std::cout << "Results        = " << results << '\n';
		//std::cout << "Recent average error = " << net.getRecentAverageError() << "\n";
		//std::cout << "Passes: " << ++i << '\n';
	}

	const auto end = std::chrono::steady_clock::now();
	const auto diff = end - start;
	std::cout << "\ttime required: " <<
		std::chrono::duration<double, std::milli>(diff).count() << '\n';

	return 0;
}
