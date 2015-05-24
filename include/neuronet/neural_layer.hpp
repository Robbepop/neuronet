#ifndef NN_NEURAL_LAYER_H
#define NN_NEURAL_LAYER_H

#include <vector>
#include <cstdint>
#include <cstddef>

#include "neuronet/neuron.hpp"

namespace neuronet {
	class NeuralNet;

	class NeuralLayer {
	public:
		enum class Kind {
			input,
			output,
			hidden
		};

		explicit NeuralLayer(NeuralNet & net, uint64_t countNeurons, Kind kind);

		//====================================================================
		// Rule-of-three
		// =============
		// Since NeuralLayer's internal members have pointers and references
		// to other internal data components custom copy and move operations
		// are required in order to ensure correct semantics.
		//====================================================================
		///*
		//NeuralLayer(const NeuralLayer & other);
		//NeuralLayer(NeuralLayer && other);
		//operator=(const NeuralLayer & rhs);
		//NeuralLayer & operator=(NeuralLayer && rhs);
		//*/

		void initializeConnections();
		void initializeBackConnections();

		      NeuralLayer & nextLayer();
		const NeuralLayer & nextLayer() const;
		      NeuralLayer & prevLayer();
		const NeuralLayer & prevLayer() const;

		void setPrevLayer(NeuralLayer & layer);
		void setNextLayer(NeuralLayer & layer);

		void feedForward();

		bool isInputLayer() const;
		bool isHiddenLayer() const;
		bool isOutputLayer() const;
		Kind getKind() const;

		//====================================================================
		// Implementing forward iterator access to internal vector
		// to enable range based for loop for instances of this class.
		//====================================================================
		std::vector<Neuron>::iterator               begin()         noexcept;
		std::vector<Neuron>::const_iterator         begin()   const noexcept;
		std::vector<Neuron>::iterator               end()           noexcept;
		std::vector<Neuron>::const_iterator         end()     const noexcept;
		std::vector<Neuron>::const_iterator         cbegin()  const noexcept;
		std::vector<Neuron>::const_iterator         cend()    const noexcept;
		std::vector<Neuron>::reverse_iterator       rbegin()        noexcept;
		std::vector<Neuron>::const_reverse_iterator rbegin()  const noexcept;
		std::vector<Neuron>::reverse_iterator       rend()          noexcept;
		std::vector<Neuron>::const_reverse_iterator rend()    const noexcept;
		std::vector<Neuron>::const_reverse_iterator crbegin() const noexcept;
		std::vector<Neuron>::const_reverse_iterator crend()   const noexcept;

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
