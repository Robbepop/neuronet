#include <memory>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "neuronet/neuron.hpp"

namespace nn {
	Neuron::Neuron(NeuralLayer * layer, double output):
		m_output{output},
		m_gradient{0.0},
		m_layer{layer}
	{}

	Neuron Neuron::createOnLayer(NeuralLayer & layer) {
		return Neuron{std::addressof(layer), 0.0};
	}

	Neuron Neuron::createBias() {
		return Neuron{nullptr, 1.0};
	}

	void Neuron::fullyConnect(NeuralLayer & layer) {
		m_connections.reserve(layer.size());
		for (auto& neuron : layer) {
			m_connections.emplace_back(*this, neuron);
		}
	}

	void Neuron::fullyConnect(std::vector<NeuralLayer> & layers) {
		auto requiredConnections = 0ul;
		for (auto& layer : layers) { requiredConnections += layer.size(); }
		m_connections.reserve(requiredConnections);
		for (auto& layer : layers) {
			fullyConnect(layer);
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
		auto sumWeights = 0.0;
		for (auto&& connection : m_inc_connections) {
			sumWeights += connection->getSource().getOutput() * connection->getWeight();
		}
		m_output = Neuron::transferFunction(sumWeights);
	}

	void Neuron::calculateOutputGradient(double targetValue) {
		assert(getLayer().isOutputLayer() &&
			"this operation is only defined for neurons within the output layer.");
		const auto delta = targetValue - m_output;
		m_gradient = delta * Neuron::transferFunctionDerivate(m_output);
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
		assert(m_layer != nullptr &&
			"this neuron is not placed within a layer; maybe it is a bias neuron?");
		return *m_layer;
	}

	auto Neuron::getLayer() const
		-> const NeuralLayer &
	{
		assert(m_layer != nullptr &&
			"this neuron is not placed within a layer; maybe it is a bias neuron?");
		return *m_layer;
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
		return 1.0 - x * x;
	}
}
