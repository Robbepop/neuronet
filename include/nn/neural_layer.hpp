#ifndef NN_NEURAL_LAYER_H
#define NN_NEURAL_LAYER_H

#include <vector>
#include <cstdint>

#include "nn/neuron.hpp"

namespace nn {
	class NeuralNet;

	class NeuralLayer {
	public:
		enum class Kind {
			input,
			output,
			hidden
		};

		explicit NeuralLayer(NeuralNet & net, uint64_t countNeurons, Kind kind);

		void initializeConnections();

		      NeuralLayer & nextLayer();
		const NeuralLayer & nextLayer() const;
		      NeuralLayer & prevLayer();
		const NeuralLayer & prevLayer() const;

		void setPrevLayer(NeuralLayer & layer);
		void setNextLayer(NeuralLayer & layer);

		      std::vector<Neuron> & neurons();
		const std::vector<Neuron> & neurons() const;

		void feedForward();

		bool isInputLayer() const;
		bool isHiddenLayer() const;
		bool isOutputLayer() const;
		Kind getKind() const;

		size_t size() const;

	private:
		NeuralLayer * m_prev_layer;
		NeuralLayer * m_next_layer;
		NeuralNet   * m_net;
		Kind          m_kind;
		std::vector<Neuron> m_neurons;
	};
}

#endif
