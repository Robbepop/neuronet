#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>

#include <iostream>

// =======================================================
// A connection instance represents a neural connection
// between two neurons.
//
// TODO: - add constructor arguments
//       - add random_weight static method
//
// Author Robin Freyler
// Date 19th May, 2015
// =======================================================

struct connection {
public:
	double weight;
	double delta_weight;
};

// =======================================================
// The neuron class represents instances of neurons within
// a layer of neuron_layers.
//
// Author Robin Freyler
// Date 19th May, 2015
// =======================================================

class neuron;
using neuron_layer = std::vector<neuron>;

class neuron {
public:
	neuron(uint32_t num_outputs, uint32_t index);
	auto set_output_val(double value) -> void;
	auto get_output_val() const -> double;
	auto feed_forward(const neuron_layer & prev_layer) -> void;
	auto calc_output_gradients(double target_value) -> void;
	auto calc_hidden_gradients(const neuron_layer & next_layer) -> void;
	auto update_input_weights(neuron_layer & prev_layer) -> void;

private:
	static double eta; // [0.0 .. 1.0] overall net training rate
	static double alpha; // [0.0 .. n] multiplier of last weight change (momentum)
	
	static auto random_weight() {
		return rand() / double(RAND_MAX);
	}
	static auto transfer_function(double x) -> double;
	static auto transfer_function_derivate(double x) -> double;

	auto sum_dow(const neuron_layer & next_layer) const -> double;

	double m_output;
	uint32_t m_index;
	double m_gradient;
	std::vector<connection> m_connections;
};

double neuron::eta = 0.15;
double neuron::alpha = 0.5;

auto neuron::set_output_val(double value) -> void {
	m_output = value;
}

auto neuron::get_output_val() const -> double {
	return m_output;
}

auto neuron::update_input_weights(neuron_layer & prev_layer) -> void {
	// The weights to be updated are in the connection container
	// in the neurons in the preceding layer
	for (auto neuron_num = 0u; neuron_num < prev_layer.size(); ++neuron_num) {
		auto& prev_neuron = prev_layer[neuron_num];
		const double old_delta_weight = prev_neuron.m_connections[m_index].delta_weight;
		const double new_delta_weight =
			// Individual input, magnified by the gradient and train rate:
			eta
			* prev_neuron.get_output_val()
			* m_gradient
			// Also add momentum = a fraction of the previous delta weight
			+ alpha
			* old_delta_weight;
		prev_neuron.m_connections[m_index].delta_weight = new_delta_weight;
		prev_neuron.m_connections[m_index].weight += new_delta_weight;
	}
}

auto neuron::sum_dow(const neuron_layer & next_layer) const -> double {
	auto sum = 0.0;

	// Sum our contributions of the errors at the nodes we feed
	for (auto neuron_num = 0u; neuron_num < next_layer.size() - 1; ++neuron_num) {
		sum += m_connections[neuron_num].weight * next_layer[neuron_num].m_gradient;
	}

	return sum;
}

auto neuron::calc_hidden_gradients(const neuron_layer & next_layer) -> void {
	auto dow = sum_dow(next_layer);
	m_gradient = dow * neuron::transfer_function_derivate(m_output);
}

auto neuron::calc_output_gradients(double target_value) -> void {
	const auto delta = target_value - m_output;
	m_gradient = delta * neuron::transfer_function_derivate(m_output);
}

auto neuron::transfer_function(double x) -> double {
	// tanh - output range [-1.0 .. 1.0]
	return tanh(x);
}

auto neuron::transfer_function_derivate(double x) -> double {
	// tanh derivative
	return 1.0 - x * x;
}

neuron::neuron(uint32_t num_outputs, uint32_t index)
{
	for (auto c = 0u; c < num_outputs; ++c) {
		m_connections.push_back(connection{});
		m_connections.back().weight = random_weight();
	}
	m_index = index;
}

auto neuron::feed_forward(
	const neuron_layer & prev_layer
) -> void {
	auto sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.
	for (auto neuron_num = 0u; neuron_num < prev_layer.size(); ++neuron_num) {
		sum += prev_layer[neuron_num].m_output *
			prev_layer[neuron_num].m_connections[m_index].weight;
	}

	m_output = neuron::transfer_function(sum);
}

// =======================================================
// The neural_net is the container element for all neurons
// within a well defined topology.
//
// Author Robin Freyler
// Date 19th May, 2015
// =======================================================

class neural_net {
public:

	neural_net(const std::vector<uint32_t> & topology);
	auto feed_forward(const std::vector<double> &) -> void;
	auto back_propagation(const std::vector<double> &) -> void;
	auto get_results(std::vector<double> &) const -> void;

private:
	std::vector<neuron_layer> m_layers;
	double m_error;
	double m_recent_avg_error;
	double m_recent_avg_smoothing_factor;
};

auto neural_net::get_results(std::vector<double> & result_values) const
	-> void
{
	result_values.clear();
	for (auto neuron_num = 0u; neuron_num < m_layers.back().size() - 1; ++neuron_num) {
		result_values.push_back(m_layers.back()[neuron_num].get_output_val());
	}
}

neural_net::neural_net(
	const std::vector<uint32_t> & topology
){
	const auto num_layers = topology.size();
	for (auto layer_num = 0u; layer_num < num_layers; ++layer_num) {
		m_layers.push_back(neuron_layer{});
		const auto num_outputs = layer_num == num_layers - 1 ? 0 : topology[layer_num + 1];

		// We have a new layer, now fill it with neurons,
		// and add a bias neuron in each layer.
		for (auto neuron_num = 0u; neuron_num <= topology[layer_num]; ++neuron_num) {
			m_layers.back().push_back(neuron{num_outputs, neuron_num});
			std::cout << "Made a neuron!\n";
		}

		// Force the bias node's output value to 1.0.
		// It's the last neuron created above.
		m_layers.back().back().set_output_val(1.0);
	}
}

auto neural_net::feed_forward(
	const std::vector<double> & input_values
) -> void {
	assert(input_values.size() == m_layers[0].size() - 1);

	// Assign (latch) the input value into the input neurons
	for (auto i = 0u; i < input_values.size(); ++i) {
		m_layers[0][i].set_output_val(input_values[i]);
	}

	// Forward propagate
	for (auto layer_num = 1u; layer_num < m_layers.size(); ++layer_num) {
		auto& prev_layer = m_layers[layer_num - 1];
		for (auto neuron_num = 0u; neuron_num < m_layers[layer_num].size() - 1; ++neuron_num) {
			m_layers[layer_num][neuron_num].feed_forward(prev_layer);
		}
	}
}

auto neural_net::back_propagation(
	const std::vector<double> & target_values
) -> void {
	// Calculate overall net error (RMS "Root Mean Square" error of output neuron errors)
	auto& output_layer = m_layers.back();
	m_error = 0.0;

	for (auto neuron_num = 0u; neuron_num < output_layer.size() - 1; ++neuron_num) {
		const auto delta = target_values[neuron_num] - output_layer[neuron_num].get_output_val();
		m_error += delta * delta;
	}
	m_error /= output_layer.size() - 1; // get average error squared
	m_error = sqrt(m_error); // RMS

	// Implement a recent average measurement:
	m_recent_avg_error =
		(m_recent_avg_error * m_recent_avg_smoothing_factor + m_error)
		/ (m_recent_avg_smoothing_factor + 1.0);

	// Calculate output layer gradients
	for (auto neuron_num = 0u; neuron_num < output_layer.size() - 1; ++neuron_num) {
		output_layer[neuron_num].calc_output_gradients(target_values[neuron_num]);
	}

	// Calculate gradients on hidden layers
	for (auto layer_num = m_layers.size() - 2; layer_num > 0; --layer_num) {
		auto& hidden_layer = m_layers[layer_num];
		auto& next_layer = m_layers[layer_num + 1];
		for (auto neuron_num = 0u; neuron_num < hidden_layer.size(); ++neuron_num) {
			hidden_layer[neuron_num].calc_hidden_gradients(next_layer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for (auto layer_num = m_layers.size() - 1; layer_num > 0; --layer_num) {
		auto& layer = m_layers[layer_num];
		auto& prev_layer = m_layers[layer_num - 1];
		for (auto neuron_num = 0u; neuron_num < layer.size(); ++neuron_num) {
			layer[neuron_num].update_input_weights(prev_layer);
		}
	}
}

// =======================================================
// Code is tested via this part.
// =======================================================

auto main() -> int {
	

	auto topology = std::vector<uint32_t>{2, 4, 1};
	auto net = neural_net{topology};

	auto input_values  = std::vector<double>{1.0, 0.0};
	auto target_values = std::vector<double>{1.0};
	auto result_values = std::vector<double>(1);

	net.feed_forward(input_values);
	//net.back_propagation(target_values);
	net.get_results(result_values);

	std::cout << result_values[0];
}
