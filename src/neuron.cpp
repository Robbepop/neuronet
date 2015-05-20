#include <memory>
#include <cassert>
#include <cmath>

#include <iostream>

#include "nn/neuron.hpp"

namespace nn {
	Neuron::Neuron(NeuralLayer & layer, Kind kind):
		m_output{kind == Neuron::Kind::bias ? 1.0 : 0.0},
		m_gradient{0.0},
		m_kind{kind},
		m_layer{std::addressof(layer)}
	{
		//std::cout << "Neuron[" << getLayer().size() << "]::created\n";
	}

	void Neuron::initializeConnections() {
		if (!getLayer().isOutputLayer()) {
			for (auto& neuron : getLayer().nextLayer().neurons()) {
				m_connections.emplace_back(*this, neuron);
			}
		}
	}

	void Neuron::setOutput(double value) {
		m_output = value;
	}

	auto Neuron::getOutput() const
		-> double
	{
		return m_output;
	}

	void Neuron::feedForward() {
		if (m_kind == Neuron::Kind::normal) {
			auto sum = 0.0;
			for (auto&& connection : m_inc_connections) {
				sum += connection->getSource().getOutput() * connection->getWeight();
			}
			m_output = Neuron::transferFunction(sum);
		}
	}

	void Neuron::calcOutputGradients(double targetValue) {
		if (m_kind == Neuron::Kind::normal) {
			const auto delta = targetValue - m_output;
			m_gradient = delta * Neuron::transferFunctionDerivate(m_output);
		}
	}

	void Neuron::calcHiddenGradients() {
		assert(getLayer().isHiddenLayer() &&
			"the layer of this neuron has to be hidden for this operation.");
		const auto dow = sumDeltaOutputWeights();
		m_gradient = dow * Neuron::transferFunctionDerivate(m_output);
	}
	
	void Neuron::updateInputWeights() {
		for (auto&& connection : m_inc_connections) {
			const double old_delta_weight = connection->getDeltaWeight();
			const double new_delta_weight =
				// Individual input, megnified by the gradient and train rate
				eta
				* connection->getSource().getOutput()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* old_delta_weight;
			connection->setDeltaWeight(new_delta_weight);
			connection->getWeight() += new_delta_weight;
		}
	}

	auto Neuron::getLayer()
		-> NeuralLayer &
	{
		return *m_layer;
	}

	auto Neuron::getLayer() const
		-> const NeuralLayer &
	{
		return *m_layer;
	}

	auto Neuron::getKind() const
		-> Kind
	{
		return m_kind;
	}

	void Neuron::registerIncConnection(NeuralConnection & connection) {
		m_inc_connections.push_back(std::addressof(connection));
	}

	double Neuron::eta   = 0.15;
	double Neuron::alpha = 0.5;

	auto Neuron::transferFunction(double x)
		-> double
	{
		return std::tanh(x);
	}

	auto Neuron::transferFunctionDerivate(double x)
		-> double
	{
		assert(x >= 0.0 && x <= 1.0 && "x is an invalid value!");
		return 1.0 - x * x;
	}

	auto Neuron::sumDeltaOutputWeights() const
		-> double
	{
		auto sum = 0.0;
		for (auto& connection : m_connections) {
			sum += connection.getWeight() * connection.getTarget().m_gradient;
		}
		return sum;
	}
}
