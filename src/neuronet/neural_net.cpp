#include <cmath>
#include <cassert>

#include "utility/reverse_adapter.hpp"
#include "utility/zip_range.hpp"

#include "neuronet/neural_net.hpp"
#include "neuronet/neural_layer.hpp"

namespace neuronet {
	NeuralNet::NeuralNet(const std::vector<uint64_t> & neuronsPerLayer):
		m_error{0.0},
		m_recent_avg_error{0.0},
		m_recent_avg_smoothing_factor{0.0},
		m_bias{Neuron::createBias()}
	{
		assert(neuronsPerLayer.size() >= 2 &&
			"there need to be a minimum of two layers in a neural network.");
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

	void NeuralNet::initializeBiasConnection() {
		for (auto& layer : m_layers) {
			m_bias.fullyConnect(layer);
		}
	}

	void NeuralNet::initializeBackConnections() {
		for (auto& layer : m_layers) {
			layer.initializeBackConnections();
		}
		m_bias.initializeBackConnections();
	}

	void NeuralNet::initializeLayers() {
		initializeLayersAdjacency();
		initializeLayersConnections();
		initializeBiasConnection();
		initializeBackConnections();
	}

	void NeuralNet::setInput(const std::vector<double> & inputValues) {
		assert(inputValues.size() == getInputLayer().size() &&
			"inputValues must have the same size as the input layer of this neural network.");
		/*
		auto inputIt = inputValues.begin();
		for (auto& neuron : getInputLayer()) {
			neuron.setOutput(*inputIt);
			++inputIt;
		}
		*/
		for (auto&& zipped : utility::zip_range(getInputLayer(), inputValues)) {
			zipped.get<0>().setOutput(zipped.get<1>());
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
		assert(targetValues.size() == getOutputLayer().size() &&
			"there must be equally many target values as neurons in the output layer.");
		m_error = 0.0;
		/*
		auto targetValuesIt = targetValues.begin();
		for (auto& neuron : getOutputLayer()) {
			const auto delta = *targetValuesIt - neuron.getOutput();
			m_error += delta * delta;
			++targetValuesIt;
		}
		*/
		for (auto&& zipped : utility::zip_range(getOutputLayer(), targetValues)) {
			const auto delta = zipped.get<1>() - zipped.get<0>().getOutput();
			m_error += delta * delta;
		}
		m_error /= getOutputLayer().size();
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
		assert(targetValues.size() == getOutputLayer().size() &&
			"there must be equally many target values as neurons in the output layer.");
		for (auto&& zipped : utility::zip_range(getOutputLayer(), targetValues)) {
			zipped.get<0>().calculateOutputGradient(zipped.get<1>());
		}
	}

	void NeuralNet::calculateHiddenLayerGradients() {
		for (auto& layer : utility::make_reverse(m_layers)) {
			if (layer.isHiddenLayer()) {
				for (auto& neuron : layer) {
					neuron.calculateHiddenGradient();
				}
			}
		}
	}

	void NeuralNet::updateConnectionWeights() {
		for (auto& layer : utility::make_reverse(m_layers)) {
			if (!layer.isInputLayer()) {
				for (auto& neuron : layer) {
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
		for (auto& neuron : getOutputLayer()) {
			result.push_back(neuron.getOutput());
		}
		return result;
	}

	auto NeuralNet::getRecentAverageError() const
		-> double
	{
		return m_recent_avg_error;
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
