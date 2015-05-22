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
		enum class Kind {
			normal,
			bias
		};

		explicit Neuron(NeuralLayer & layer, Kind kind);

		void initializeConnections();

		void setOutput(double value);
		auto getOutput() const -> double;

		void feedForward();
		void calcOutputGradients(double targetValue);
		void calcHiddenGradients();
		void updateInputWeights();

		      NeuralLayer & getLayer();
		const NeuralLayer & getLayer() const;

		Kind getKind() const;

		void registerIncConnection(NeuralConnection & connection);

	private:
		static double eta;   // [0 .. 1] overall net training rate
		static double alpha; // [0 .. n] multiplier of last weight change (momentum)

		static auto transferFunction(double x) -> double;
		static auto transferFunctionDerivate(double x) -> double;

		auto sumDeltaOutputWeights() const -> double;

		double m_output;
		double m_gradient;
		Kind m_kind;
		NeuralLayer * m_layer;
		std::vector<NeuralConnection > m_connections;
		std::vector<NeuralConnection*> m_inc_connections;
	};
}

#endif
