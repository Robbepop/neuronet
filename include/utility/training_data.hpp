#ifndef NN_TRAINING_DATA_H
#define NN_TRAINING_DATA_H

#include <string>
#include <vector>
#include <cstdint>

namespace utility {
	class TrainingPass {
	public:
		explicit TrainingPass(
			std::vector<double> inputValues,
			std::vector<double> expectedValues);

		auto getInputValues()    const -> const std::vector<double> &;
		auto getExpectedValues() const -> const std::vector<double> &;

	private:
		std::vector<double> m_input;
		std::vector<double> m_expected;
	};

	class TrainingData {
	public:
		//====================================================================
		// Constructs a new instance of TrainingData by reading in the
		// file at the given path with a specified format.
		//
		// -------------------------------------------------------------------
		// The format for a file is as follows:
		//
		// ===================================================================
		// topology n1 n2 ... nx
		//
		// input    i1 i2 ... iy
		// expected e1 e2 ... ez
		//
		// input    i1 i2 ... iy
		// expected e1 e2 ... ez
		//
		// ...
		//
		// input    i1 i2 ... iy
		// expected e1 e2 ... ez
		// ===================================================================
		//
		// Where the content after 'topology = ...' stores the information
		// about how many neurons there are in the several neuronal layers.
		// The amount of such numbers x stands for the amount of different
		// layers in the neuronal network for this test data.
		//
		// The passes all have y input
		// values specified the line after the 'input' keyword and z
		// expected values specified in the line after the 'expected' keyword.
		// -------------------------------------------------------------------
		//
		// This function will throw an expection if:
		//     - the file specified in the 'path' doesn't met the format
		//       requirements.
		//     - the amount of input values isn't the same as the specified
		//       amount of neurons in the input layer.
		//     - the amount of expected values isn't the same as the specified
		//       amount of neurons in the output layer.
		//====================================================================
		explicit TrainingData(const std::string & pathToData);

		auto getTopology() const -> const std::vector<uint64_t> &;

		//====================================================================
		// Implementing forward iterator access to internal vector
		// to enable range based for loop for instances of this class.
		// Later can be extended with const and reverse iterator accesses.
		//====================================================================
		std::vector<TrainingPass>::iterator               begin()         noexcept;
		std::vector<TrainingPass>::const_iterator         begin()   const noexcept;
		std::vector<TrainingPass>::iterator               end()           noexcept;
		std::vector<TrainingPass>::const_iterator         end()     const noexcept;
		std::vector<TrainingPass>::const_iterator         cbegin()  const noexcept;
		std::vector<TrainingPass>::const_iterator         cend()    const noexcept;
		std::vector<TrainingPass>::reverse_iterator       rbegin()        noexcept;
		std::vector<TrainingPass>::const_reverse_iterator rbegin()  const noexcept;
		std::vector<TrainingPass>::reverse_iterator       rend()          noexcept;
		std::vector<TrainingPass>::const_reverse_iterator rend()    const noexcept;
		std::vector<TrainingPass>::const_reverse_iterator crbegin() const noexcept;
		std::vector<TrainingPass>::const_reverse_iterator crend()   const noexcept;

	private:
		auto parseTopology(std::string & line)       -> std::vector<uint64_t>;
		auto parseInputValues(std::string & line)    -> std::vector<double>;
		auto parseExpectedValues(std::string & line) -> std::vector<double>;

		std::vector<uint64_t> m_topology;
		std::vector<TrainingPass> m_passes;
	};
}

#endif
