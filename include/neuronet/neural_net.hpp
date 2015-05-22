#ifndef NN_NEURAL_NET_H
#define NN_NEURAL_NET_H

#include <vector>
#include <cstdint>

namespace nn {
	class NeuralLayer;

	//========================================================
	// This class represents a neural network working with
	// feedForward as well as backPropagation and gradient
	// approach learning.
	// Can be feed with input values and used to compute
	// and learn calculations.
	//
	// Author  Robin Freyler
	// Date    20th May, 2015
	//========================================================
	class NeuralNet {
	public:
		//========================================================
		// Creates a new instance of a neural net with the given
		// amount of layers and the given amount of neurons per
		// layer.
		//========================================================
		explicit NeuralNet(const std::vector<uint64_t> & neurons_per_layer);

		//====================================================================
		// Rule-of-three
		// =============
		// Since NeuralLayer's internal members have pointers and references
		// to other internal data components custom copy and move operations
		// are required in order to ensure correct semantics.
		//====================================================================
		/*
		NeuralLayer(const NeuralLayer & other);
		NeuralLayer(NeuralLayer && other);
		operator=(const NeuralLayer & rhs);
		operator=(NeuralLayer && rhs);
		*/

		// The neural network takes the input values and computes
		// their values with its current state.
		// results() can be used to read the result of this
		// computation.
		void feedForward(const std::vector<double> & inputValues);

		// Used to make this neural network adapt and learn
		// with expected values given as parameters.
		void backPropagation(const std::vector<double> & targetValues);

		// Returns results in the output values of the Output Layer
		// of the latest computation of feedForward and/or backPropagation.
		auto results() const -> std::vector<double>;

		auto getRecentAverageError() const -> double;

	private:
		//========================================================
		// These are helper functions to improve code readability
		// while accessing the input layer of a neural network.
		//========================================================
		auto getInputLayer()       ->       NeuralLayer &;
		auto getInputLayer() const -> const NeuralLayer &;

		//========================================================
		// These are helper functions to improve code readability
		// while accessing the output layer of a neural network.
		//========================================================
		auto getOutputLayer()       ->       NeuralLayer &;
		auto getOutputLayer() const -> const NeuralLayer &;

		//========================================================
		// These private helper functions are used to improve
		// code readability for the initialization parts to make
		// it more clear when has what to be initialized.
		//========================================================
		void initializeLayersAdjacency();
		void initializeLayersConnections();
		void initializeLayers();

		//========================================================
		// These private helper functions are mainly used to
		// break down the huge back propagation function into
		// several minor logical pieces of code.
		//========================================================
		void calculateOverallNetError(const std::vector<double> & targetValues);
		void calculateAverageError();
		void calculateOutputLayerGradients(const std::vector<double> & targetValues);
		void calculateHiddenLayerGradients();
		void updateConnectionWeights();

		//========================================================
		// This is used by the feedForward method in order to
		// set the input values within the input layer.
		// Note: Maybe this method is also helpful as public.
		//========================================================
		void setInput(const std::vector<double> &);

		//========================================================
		// Private Members
		// ===============
		//   m_error
		//   m_recent_avg_error
		//   m_recent_avg_smoothing_factor
		//   m_layers - stores the layers of this neural net
		//========================================================
		double m_error;
		double m_recent_avg_error;
		double m_recent_avg_smoothing_factor;
		std::vector<NeuralLayer> m_layers;
	};
}

#endif
