#ifndef REVERSE_ADAPTER_H
#define REVERSE_ADAPTER_H

namespace utility {

	template <typename T>
	class reverse_adapter {
	public:
		reverse_adapter(T & container):
			m_container{container}
		{}

		typename T::reverse_iterator begin() {
			return m_container.rbegin();
		}

		typename T::reverse_iterator end() {
			return m_container.rend();
		}
	private:
		T & m_container;
	};

	template <typename T>
	class const_reverse_adapter {
	public:
		const_reverse_adapter(const T & container):
			m_container{container}
		{}

		typename T::const_reverse_iterator begin() {
			return m_container.rbegin();
		}

		typename T::const_reverse_iterator end() {
			return m_container.rend();
		}
	private:
		const T & m_container;
	};

	template <typename T>
	reverse_adapter<T> make_reverse(T & container) {
		return reverse_adapter<T>{container};
	}

	template <typename T>
	const_reverse_adapter<T> make_reverse(const T & container) {
		return const_reverse_adapter<T>{container};
	}

}

#endif
