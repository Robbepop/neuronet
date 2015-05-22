#include <cstddef>
#include <cassert>
#include <iostream>

#include "neuronet/neural_layer.hpp"

namespace nn {
	NeuralLayer::NeuralLayer(NeuralNet & net, uint64_t countNeurons, NeuralLayer::Kind kind):
		m_prev_layer{nullptr},
		m_next_layer{nullptr},
		m_net{std::addressof(net)},
		m_kind{kind}
	{
		m_neurons.reserve(countNeurons);
		while (m_neurons.size() < countNeurons) {
			m_neurons.emplace_back(*this, Neuron::Kind::normal);
		}
		m_neurons.emplace_back(*this, Neuron::Kind::bias);
		std::cout << "NeuralLayer[" << size() << "]::created\n";
	}

	void NeuralLayer::initializeConnections() {
		for (auto& neuron : m_neurons) {
			neuron.initializeConnections();
		}
	}

	auto NeuralLayer::nextLayer()
		-> NeuralLayer &
	{
		assert(!isOutputLayer() &&
			"can't get the next layer of the output layer.");
		return *m_next_layer;
	}

	auto NeuralLayer::nextLayer() const
		-> const NeuralLayer &
	{
		assert(isOutputLayer() &&
			"can't get the next layer of the output layer.");
		return *m_next_layer;
	}

	auto NeuralLayer::prevLayer()
		-> NeuralLayer &
	{
		assert(isInputLayer() &&
			"can't get the previous layer of the input layer.");
		return *m_prev_layer;
	}

	auto NeuralLayer::prevLayer() const
		-> const NeuralLayer &
	{
		assert(isInputLayer() &&
			"can't get the previous layer of the input layer.");
		return *m_prev_layer;
	}

	void NeuralLayer::setPrevLayer(NeuralLayer & layer) {
		m_prev_layer = std::addressof(layer);
	}

	void NeuralLayer::setNextLayer(NeuralLayer & layer) {
		m_next_layer = std::addressof(layer);
	}

	void NeuralLayer::feedForward() {
		std::cout << "NeuralLayer[" << size() << "]::feedForward start\n";
		if (!isInputLayer()) {
			std::cout << "NeuralLayer[" << size() << "]::feedForward no InputLayer start\n";
			for (auto& neuron : m_neurons) {
				neuron.feedForward();
			}
			std::cout << "NeuralLayer[" << size() << "]::feedForward no InputLayer end\n";
		}
		std::cout << "NeuralLayer[" << size() << "]::feedForward end\n";
	}

	bool NeuralLayer::isInputLayer() const {
		return m_kind == NeuralLayer::Kind::input;
	}

	bool NeuralLayer::isHiddenLayer() const {	
		return m_kind == NeuralLayer::Kind::hidden;
	}

	bool NeuralLayer::isOutputLayer() const {
		return m_kind == NeuralLayer::Kind::output;
	}

	auto NeuralLayer::getKind() const
		-> NeuralLayer::Kind
	{
		return m_kind;
	}

	auto NeuralLayer::size() const
		-> size_t
	{
		return m_neurons.size();
	}

	//=========================================================================
	// Iterator Wrappers
	//=======================================================================

	auto NeuralLayer::begin() noexcept
		-> decltype(m_neurons.begin())
	{
		return m_neurons.begin();
	}

	auto NeuralLayer::begin() const noexcept
		-> decltype(m_neurons.begin())
	{
		return m_neurons.begin();
	}

	auto NeuralLayer::end() noexcept
		-> decltype(m_neurons.end())
	{
		return m_neurons.end();
	}

	auto NeuralLayer::end() const noexcept
		-> decltype(m_neurons.end())
	{
		return m_neurons.end();
	}

	auto NeuralLayer::cbegin() const noexcept
		-> decltype(m_neurons.cbegin())
	{
		return m_neurons.cbegin();
	}

	auto NeuralLayer::cend() const noexcept
		-> decltype(m_neurons.cend())
	{
		return m_neurons.cend();
	}

	auto NeuralLayer::rbegin() noexcept
		-> decltype(m_neurons.rbegin())
	{
		return m_neurons.rbegin();
	}

	auto NeuralLayer::rbegin() const noexcept
		-> decltype(m_neurons.rbegin())
	{
		return m_neurons.rbegin();
	}

	auto NeuralLayer::rend() noexcept
		-> decltype(m_neurons.rend())
	{
		return m_neurons.rend();
	}

	auto NeuralLayer::rend() const noexcept
		-> decltype(m_neurons.rend())
	{
		return m_neurons.rend();
	}

	auto NeuralLayer::crbegin() const noexcept
		-> decltype(m_neurons.crbegin())
	{
		return m_neurons.crbegin();
	}

	auto NeuralLayer::crend() const noexcept
		-> decltype(m_neurons.crend())
	{
		return m_neurons.crend();
	}
}
