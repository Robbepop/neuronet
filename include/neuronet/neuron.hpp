#ifndef NN_NEURON_H
#define NN_NEURON_H

#include <vector>
#include <cstdint>

#include "neuronet/neural_connection.hpp"
#include "neuronet/neural_layer.hpp"

namespace nn {
	class NeuralLayer;

	class Neuron {
	public:
		Neuron static createOnLayer(NeuralLayer & layer);
		Neuron static createBias();

		//void initializeConnections();
		void initializeConnections(NeuralLayer & layer);
		void initializeConnections(std::vector<NeuralLayer> & layers);

		void setOutput(double value);
		auto getOutput() const -> double;

		void feedForward();
		void calculateOutputGradient(double targetValue);
		void calculateHiddenGradient();
		void updateInputWeights();

		void registerIncConnection(NeuralConnection & connection);

	private:
		explicit Neuron(NeuralLayer * layer, double output);

		      NeuralLayer & getLayer();
		const NeuralLayer & getLayer() const;

		static double eta;   // [0 .. 1] overall net training rate
		static double alpha; // [0 .. n] multiplier of last weight change (momentum)

		static auto transferFunction(double x) -> double;
		static auto transferFunctionDerivate(double x) -> double;

		auto sumDeltaOutputWeights() const -> double;

		double m_output;
		double m_gradient;
		NeuralLayer * m_layer;
		std::vector<NeuralConnection > m_connections;
		std::vector<NeuralConnection*> m_inc_connections;
	};
}

#endif
