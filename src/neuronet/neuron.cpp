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
	{
		std::cout << "Neuron[" << getLayer().size() << "]::created\n";
	}

	void Neuron::initializeConnections() {
		assert(!getLayer().isOutputLayer() &&
			"can't initialize connections of neurons in the output layer!");
		//if (!getLayer().isOutputLayer()) {
		m_connections.reserve(getLayer().nextLayer().size() - 1); // without bias
			for (auto& neuron : getLayer().nextLayer()) {
				if (neuron.getKind() == Neuron::Kind::normal) {
					m_connections.emplace_back(*this, neuron);
				}
			}
		//}
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
		//std::cout << "Neuron[" << getLayer().size() << "]::feedForward start\n";
		//std::cout << "Neuron[" << getLayer().size() <<
		//	"]::feedForward countIncoming = " << m_inc_connections.size() << '\n';
		if (getKind() == Neuron::Kind::normal) {
			//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal start\n";
			auto sumWeights = 0.0;
			//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal 1\n";
			for (auto& connection : m_inc_connections) {
				//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal 2a\n";
				sumWeights += connection->getSource().getOutput() * connection->getWeight();
				//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal 2b\n";
			}
			//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal 3\n";
			m_output = Neuron::transferFunction(sumWeights);
			//std::cout << "Neuron[" << getLayer().size() << "]::feedForward isNormal end\n";
		}
		//std::cout << "Neuron[" << getLayer().size() << "]::feedForward end\n";
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
	
	void Neuron::updateInputWeights() {
		assert(!getLayer().isInputLayer() &&
			"this operation is not defined for neurons within the input layer.");
		//std::cout << "Neuron[" << getLayer().size() << "]::updateInputWeights start\n";
		for (auto& connection : m_inc_connections) {
			//auto a = connection->getSource().getOutput();
			//std::cout << "Neuron[" << getLayer().size() << "]::updateInputWeights loop 1\n";
			const double newDeltaWeight =
				// Individual input, megnified by the gradient and train rate
				eta
				* connection->getSource().getOutput()
				* m_gradient
				// Also add momentum = a fraction of the previous delta weight
				+ alpha
				* connection->getDeltaWeight();
			//std::cout << "Neuron[" << getLayer().size() << "]::updateInputWeights loop 2\n";
			connection->setWeight(connection->getWeight() + newDeltaWeight);
			//std::cout << "Neuron[" << getLayer().size() << "]::updateInputWeights loop 3\n";
		}
		//std::cout << "Neuron[" << getLayer().size() << "]::updateInputWeights end\n";
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
