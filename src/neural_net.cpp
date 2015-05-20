#include <cmath>
#include <iterator>
#include <cassert>
#include <iostream>

#include "nn/neural_net.hpp"
#include "nn/neural_layer.hpp"

namespace nn {
	NeuralNet::NeuralNet(const std::vector<uint64_t> & neuronsPerLayer):
		m_error{0.0},
		m_recent_avg_error{0.0},
		m_recent_avg_smoothing_factor{0.0}
	{
		m_layers.reserve(neuronsPerLayer.size());
		for (auto countNeurons : neuronsPerLayer) {
			const auto layerKind =
				m_layers.empty()                              ? NeuralLayer::Kind::input :
				m_layers.size() == neuronsPerLayer.size() - 1 ? NeuralLayer::Kind::output :
				                                                NeuralLayer::Kind::hidden;
			m_layers.emplace_back(*this, countNeurons, layerKind);
		}
		initializeLayers();
	}

	void NeuralNet::initializeLayersAdjacency() {
		NeuralLayer * previous = nullptr;
		for (auto& layer : m_layers) {
			if (!layer.isInputLayer()) {
				layer.setPrevLayer(*previous);
				previous->setNextLayer(layer);
			}
			previous = &layer;
		}
	}

	void NeuralNet::initializeLayersConnections() {
		for (auto& layer : m_layers) {
			layer.initializeConnections();
		}
	}

	void NeuralNet::initializeLayers() {
		initializeLayersAdjacency();
		initializeLayersConnections();
	}

	void NeuralNet::setInput(const std::vector<double> & inputValues) {
		auto currentNeuron = getInputLayer().neurons().begin();
		for (auto input : inputValues) {
			currentNeuron->setOutput(input);
			++currentNeuron;
		}
	}

	void NeuralNet::feedForward(const std::vector<double> & inputValues) {
		assert(inputValues.size() == getInputLayer().size() &&
			"inputValues must have the same size as the input layer of this neural network.");

		setInput(inputValues);
		for (auto& layer : m_layers) {
			layer.feedForward();
		}
	}

	void NeuralNet::calculateOverallNetError(
		const std::vector<double> & targetValues
	) {
		m_error = 0.0;
		auto targetValuesIt = targetValues.begin();
		for (auto& neuron : getOutputLayer().neurons()) {
			if (neuron.getKind() != Neuron::Kind::bias) {
				const auto delta = *targetValuesIt - neuron.getOutput();
				m_error += delta * delta;
				++targetValuesIt;
			}
		}
		m_error /= getOutputLayer().size() - 1;
		m_error = std::sqrt(m_error);
	}

	void NeuralNet::calculateAverageError() {
		m_recent_avg_error =
			(m_recent_avg_error * m_recent_avg_smoothing_factor + m_error)
			/ (m_recent_avg_smoothing_factor + 1.0);
	}

	void NeuralNet::calculateOutputLayerGradients(
		const std::vector<double> & targetValues
	) {
		auto neuronIt = getOutputLayer().neurons().begin();
		for (auto value : targetValues) {
			neuronIt->calcOutputGradients(value);
			++neuronIt;
		}
	}

	void NeuralNet::calculateHiddenLayerGradients() {
		for (auto& layer : m_layers) {
			if (layer.isHiddenLayer()) {
				for (auto& neuron : layer.neurons()) {
					neuron.calcHiddenGradients();
				}
			}
		}
	}

	void NeuralNet::updateConnectionWeights() {
		/*
		for (auto& layer : std::make_reverse_iterator(m_layers)) {
			if (!layer.isInputLayer()) {
				for (auto& neuron : layer.neurons()) {
					neuron.updateInputWeights();
				}
			}
		}
		*/
		for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
			auto& layer = *it;
			if (!layer.isInputLayer()) {
				for (auto& neuron : layer.neurons()) {
					neuron.updateInputWeights();
				}
			}
		}
	}

	void NeuralNet::backPropagation(const std::vector<double> & targetValues) {
		assert(targetValues.size() == getOutputLayer().size() &&
			"targetValues must have the same size as the output layer of this neural network.");
		calculateOverallNetError(targetValues);
		calculateAverageError();
		calculateOutputLayerGradients(targetValues);
		calculateHiddenLayerGradients();
		updateConnectionWeights();
	}

	auto NeuralNet::results() const
		-> std::vector<double>
	{
		auto result = std::vector<double>{};
		     result.reserve(getOutputLayer().size());
		for (auto& neuron : m_layers.back().neurons()) {
			result.push_back(neuron.getOutput());
		}
		return result;
	}

	auto NeuralNet::getInputLayer()
		-> NeuralLayer &
	{
		return m_layers.front();
	}

	auto NeuralNet::getInputLayer() const
		-> const NeuralLayer &
	{
		return m_layers.front();
	}

	auto NeuralNet::getOutputLayer()
		-> NeuralLayer &
	{
		return m_layers.back();
	}

	auto NeuralNet::getOutputLayer() const
		-> const NeuralLayer &
	{
		return m_layers.back();
	}
}
