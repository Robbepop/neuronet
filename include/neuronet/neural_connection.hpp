#ifndef NN_NEURAL_CONNECTION_H
#define NN_NEURAL_CONNECTION_H

#include <cstdint>

namespace nn {
	class Neuron;

	class NeuralConnection {
	public:
		explicit NeuralConnection(Neuron & source, Neuron & target);

		void setWeight(double value);
		auto getWeight() const -> double;
		auto getDeltaWeight() const -> double;

		      Neuron & getSource();
		const Neuron & getSource() const;

		      Neuron & getTarget();
		const Neuron & getTarget() const;

	private:
		static auto randomWeight() -> double;

		double m_weight;
		double m_delta_weight;
		Neuron * m_source;
		Neuron * m_target;
	};
}

#endif
