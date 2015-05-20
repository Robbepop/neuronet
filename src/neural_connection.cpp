#include <memory>
#include <random>

#include "nn/neuron.hpp"
#include "nn/neural_connection.hpp"

namespace nn {
	NeuralConnection::NeuralConnection(
		Neuron & source, Neuron & target
	):
		m_weight{randomWeight()},
		m_delta_weight{0.0},
		m_source{std::addressof(source)},
		m_target{std::addressof(target)}
	{
		target.registerIncConnection(*this);
	}

	void NeuralConnection::setWeight(double value) { m_weight = value; }
	auto NeuralConnection::getWeight() const
		-> double
	{
		return m_weight;
	}
	auto NeuralConnection::getWeight()
		-> double &
	{
		return m_weight;
	}

	void NeuralConnection::setDeltaWeight(double value) { m_delta_weight = value; }
	auto NeuralConnection::getDeltaWeight() const
		-> double
	{
		return m_delta_weight;
	}
	auto NeuralConnection::getDeltaWeight()
		-> double &
	{
		return m_delta_weight;
	}

	auto NeuralConnection::getSource()
		-> Neuron &
	{
		return *m_source;
	}

	auto NeuralConnection::getSource() const
		-> const Neuron &
	{
		return *m_source;
	}

	auto NeuralConnection::getTarget()
		-> Neuron &
	{
		return *m_target;
	}

	auto NeuralConnection::getTarget() const
		-> const Neuron &
	{
		return *m_target;
	}

	auto NeuralConnection::randomWeight() -> double {
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::uniform_real_distribution<> dis(0.0, 1.0);
		return dis(gen);
	}

}
