#ifndef PRINT_VECTOR_H
#define PRINT_VECTOR_H

#include <iterator>
#include <iostream>

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	if ( !v.empty() ) {
		out << '[';
		std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
		out << "\b\b]";
	}
	return out;
}

#endif
