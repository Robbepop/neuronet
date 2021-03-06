#include <cstddef>
#include <cassert>

#include "neuronet/neural_layer.hpp"

namespace neuronet {
	NeuralLayer::NeuralLayer(NeuralNet & net, uint64_t countNeurons, NeuralLayer::Kind kind):
		m_prev_layer{nullptr},
		m_next_layer{nullptr},
		m_net{std::addressof(net)},
		m_kind{kind}
	{
		assert(countNeurons >= 1 &&
			"there must be a minimum of one neuron in any neural layer.");
		m_neurons.reserve(countNeurons);
		while (m_neurons.size() < countNeurons) {
			m_neurons.push_back(Neuron::createOnLayer(*this));
		}
	}

	void NeuralLayer::initializeConnections() {
		if (!isOutputLayer())
			// Connections are always established from the current
			// neural layer to the next layer. Since the ouput layer
			// shouldn't have a next layer there are no connections
			// to initialize.
		{
			for (auto& neuron : m_neurons) {
				neuron.fullyConnect(nextLayer());
			}
		}
	}

	void NeuralLayer::initializeBackConnections() {
		if (!isOutputLayer()) {
			for (auto& neuron : m_neurons) {
				neuron.initializeBackConnections();
			}
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
		if (!isInputLayer()) {
			for (auto& neuron : m_neurons) {
				neuron.feedForward();
			}
		}
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
	//=========================================================================

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
