#include <memory>
#include <cassert>
#include <cmath>

#include <iostream>

#include "neuronet/neuron.hpp"

namespace nn {
	Neuron::Neuron(NeuralLayer & layer, Kind kind):
		m_output{kind == Neuron::Kind::bias ? 1.0 : 0.0},
		m_gradient{0.0},
		m_kind{kind},
		m_layer{std::addressof(layer)}
	{}

	void Neuron::initializeConnections() {
		assert(!getLayer().isOutputLayer() &&
			"can't initialize connections of neurons in the output layer!");
		m_connections.reserve(getLayer().nextLayer().size() - 1); // without bias
		for (auto& neuron : getLayer().nextLayer()) {
			if (neuron.getKind() == Neuron::Kind::normal) {
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
		if (getKind() == Neuron::Kind::normal) {
			auto sumWeights = 0.0;
			for (auto& connection : m_inc_connections) {
				sumWeights += connection->getSource().getOutput() * connection->getWeight();
			}
			m_output = Neuron::transferFunction(sumWeights);
		}
	}

	void Neuron::calculateOutputGradient(double targetValue) {
		assert(getLayer().isOutputLayer() &&
			"this operation is only defined for neurons within the output layer.");
		if (m_kind == Neuron::Kind::normal) {
			const auto delta = targetValue - m_output;
			m_gradient = delta * Neuron::transferFunctionDerivate(m_output);
		}
	}

	void Neuron::calculateHiddenGradient() {
		assert(getLayer().isHiddenLayer() &&
			"this operation is only defined for neurons within the hidden layer.");
		m_gradient = sumDeltaOutputWeights() * Neuron::transferFunctionDerivate(m_output);
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
	
	void Neuron::updateInputWeights() {
		assert(!getLayer().isInputLayer() &&
			"this operation is not defined for neurons within the input layer.");
		for (auto& connection : m_inc_connections) {
			const double newDeltaWeight =
				// Individual input, megnified by the gradient and train rate
				eta
				* connection->getSource().getOutput()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* connection->getDeltaWeight();
			connection->setWeight(connection->getWeight() + newDeltaWeight);
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

	double Neuron::eta   = 0.5;
	double Neuron::alpha = 0.5;

	auto Neuron::transferFunction(double x)
		-> double
	{
		return std::tanh(x);
	}

	auto Neuron::transferFunctionDerivate(double x)
		-> double
	{
		return 1.0 - x * x;
	}
}
