#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "utility/training_data.hpp"

namespace utility {
	//===========================================================
	// TrainingPass Implementation
	//===========================================================
	TrainingPass::TrainingPass(
		std::vector<double> inputValues,
		std::vector<double> expectedValues
	):
		m_input{std::move(inputValues)},
		m_expected{std::move(expectedValues)}
	{}

	auto TrainingPass::getInputValues() const
		-> const std::vector<double> &
	{
		return m_input;
	}

	auto TrainingPass::getExpectedValues() const
		-> const std::vector<double> &
	{
		return m_expected;
	}

	//===========================================================
	// TrainingData Implementation
	//===========================================================
	std::vector<uint64_t> TrainingData::parseTopology(std::string & line) {
		using namespace std::string_literals;
		auto parts = std::vector<std::string>{};
		boost::split(parts, line,
			boost::is_any_of("\t "),
			boost::token_compress_on);
		if (parts.front() != "topology"s) {
			throw std::invalid_argument{
				"expected keyword 'topology' at this point of the input file."};
		}
		auto topology = std::vector<uint64_t>(parts.size() - 1);
		std::transform(
			parts.begin() + 1, parts.end(), topology.begin(),
			[](const std::string & elem) {
				return std::stoul(elem);
			});
		return topology;
	}

	std::vector<double> TrainingData::parseInputValues(std::string & line) {
		using namespace std::string_literals;
		auto parts = std::vector<std::string>{};
		boost::split(parts, line,
			boost::is_any_of("\t "),
			boost::token_compress_on);
		if (parts.front() != "input"s) {
			throw std::invalid_argument{
				"expected keyword 'input' at this point of the input file."};
		}
		auto inputValues = std::vector<double>(parts.size() - 1);
		std::transform(
			parts.begin() + 1, parts.end(), inputValues.begin(),
			[](const std::string & elem) {
				return std::stod(elem);
			});
		return inputValues;
	}

	std::vector<double> TrainingData::parseExpectedValues(std::string & line) {
		using namespace std::string_literals;
		auto parts = std::vector<std::string>{};
		boost::split(parts, line,
			boost::is_any_of("\t "),
			boost::token_compress_on);
		if (parts.front() != "expected"s) {
			throw std::invalid_argument{
				"expected keyword 'expected' at this point of the input file."};
		}
		auto expectedValues = std::vector<double>(parts.size() - 1);

		std::transform(
			parts.begin() + 1, parts.end(), expectedValues.begin(),
			[](const std::string & elem) {
				return std::stod(elem);
			});
		return expectedValues;
	}

	TrainingData::TrainingData(const std::string & pathToData) {
		using namespace std::string_literals;
		std::ifstream stream{pathToData};
		auto line   = ""s;
		std::getline(stream, line);
		std::cout << line.size() << '\n';
		m_topology  = parseTopology(line);

		while (std::getline(stream, line)) {
			if (line.empty()) continue;
			auto inputValues = parseInputValues(line);
			std::getline(stream, line);
			auto expectedValues = parseExpectedValues(line);
			m_passes.emplace_back(
				std::move(inputValues),
				std::move(expectedValues));
		}
	}

	auto TrainingData::getTopology() const
		-> const std::vector<uint64_t> &
	{
		return m_topology;
	}

	//=========================================================================
	// Iterator Wrappers
	//=======================================================================

	auto TrainingData::begin() noexcept
		-> decltype(m_passes.begin())
	{
		return m_passes.begin();
	}

	auto TrainingData::begin() const noexcept
		-> decltype(m_passes.begin())
	{
		return m_passes.begin();
	}

	auto TrainingData::end() noexcept
		-> decltype(m_passes.end())
	{
		return m_passes.end();
	}

	auto TrainingData::end() const noexcept
		-> decltype(m_passes.end())
	{
		return m_passes.end();
	}

	auto TrainingData::cbegin() const noexcept
		-> decltype(m_passes.cbegin())
	{
		return m_passes.cbegin();
	}

	auto TrainingData::cend() const noexcept
		-> decltype(m_passes.cend())
	{
		return m_passes.cend();
	}

	auto TrainingData::rbegin() noexcept
		-> decltype(m_passes.rbegin())
	{
		return m_passes.rbegin();
	}

	auto TrainingData::rbegin() const noexcept
		-> decltype(m_passes.rbegin())
	{
		return m_passes.rbegin();
	}

	auto TrainingData::rend() noexcept
		-> decltype(m_passes.rend())
	{
		return m_passes.rend();
	}

	auto TrainingData::rend() const noexcept
		-> decltype(m_passes.rend())
	{
		return m_passes.rend();
	}

	auto TrainingData::crbegin() const noexcept
		-> decltype(m_passes.crbegin())
	{
		return m_passes.crbegin();
	}

	auto TrainingData::crend() const noexcept
		-> decltype(m_passes.crend())
	{
		return m_passes.crend();
	}
}
