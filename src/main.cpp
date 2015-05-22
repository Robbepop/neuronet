#include <cstddef>
#include <iostream>

#include "neuronet/neural_connection.hpp"
#include "neuronet/neural_layer.hpp"
#include "neuronet/neural_net.hpp"
#include "neuronet/neuron.hpp"

#include "utility/training_data.hpp"
#include "utility/print_vector.hpp"

int main() {
	auto data = utility::TrainingData{"../test/test_03.data"};
	auto net  = nn::NeuralNet{data.getTopology()};
	std::cout << "Input Topology = " << data.getTopology() << '\n' << '\n';
	//auto i = 0ul;
	for (auto&& pass : data) {
		//std::cout << "InputValues    = " << pass.getInputValues() << '\n';
		//std::cout << "ExpectedValues = " << pass.getExpectedValues() << '\n';
		net.feedForward(pass.getInputValues());
		net.backPropagation(pass.getExpectedValues());
		auto results = net.results();
		//std::cout << "Results        = " << results << '\n';
		std::cout << "Recent average error = " << net.getRecentAverageError() << "\n\n";
		//std::cout << "Passes: " << ++i << '\n';
	}

	return 0;
}
