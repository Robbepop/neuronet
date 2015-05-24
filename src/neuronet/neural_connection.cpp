#include <cassert>
#include <memory>
#include <random>

#include "neuronet/neuron.hpp"
#include "neuronet/neural_connection.hpp"

namespace neuronet {
	NeuralConnection::NeuralConnection(
		Neuron & source, Neuron & target
	):
		m_weight{randomWeight()},
		m_delta_weight{0.0},
		m_source{std::addressof(source)},
		m_target{std::addressof(target)}
	{}

	void NeuralConnection::setWeight(double newWeight) {
		m_delta_weight = newWeight - m_weight;
		m_weight = newWeight;
	}

	auto NeuralConnection::getWeight() const
		-> double
	{
		return m_weight;
	}

	auto NeuralConnection::getDeltaWeight() const
		-> double
	{
		return m_delta_weight;
	}

	auto NeuralConnection::getSource()
		-> Neuron &
	{
		assert(m_source != nullptr &&
			"source of this connection isn't defined!");
		return *m_source;
	}

	auto NeuralConnection::getSource() const
		-> const Neuron &
	{
		assert(m_source != nullptr &&
			"source of this connection isn't defined!");
		return *m_source;
	}

	auto NeuralConnection::getTarget()
		-> Neuron &
	{
		assert(m_source != nullptr &&
			"target of this connection isn't set!");
		return *m_target;
	}

	auto NeuralConnection::getTarget() const
		-> const Neuron &
	{
		assert(m_source != nullptr &&
			"target of this connection isn't set!");
		return *m_target;
	}

	auto NeuralConnection::randomWeight() -> double {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_real_distribution<> dis(0.0, 1.0);
		return dis(gen);
	}

}
