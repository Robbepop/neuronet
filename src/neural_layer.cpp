#include <cstddef>
#include <cassert>
#include <iostream>

#include "nn/neural_layer.hpp"

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
		//std::cout << "NeuralLayer[" << size() << "]::created\n";
	}

	void NeuralLayer::initializeConnections() {
		for (auto& neuron : neurons()) {
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

	auto NeuralLayer::neurons()
		-> std::vector<Neuron> &
	{
		return m_neurons;
	}

	auto NeuralLayer::neurons() const
		-> const std::vector<Neuron> &
	{
		return m_neurons;
	}

	void NeuralLayer::feedForward() {
		if (!isInputLayer()) {
			for (auto& neuron : neurons()) {
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
}
