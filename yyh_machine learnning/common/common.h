#pragma once

#include <string>
#include <vector>

#pragma warning(disable:4244 4081);

namespace common{

template<typename T>
T exchange(std::string& src)
{
	return atoll(src.c_str());
}
template<>
inline double exchange(std::string& src)
{
	return atof(src.c_str());
}
template<>
inline float exchange(std::string& src)
{
	return float(atof(src.c_str()));
}
template<>
inline std::string exchange(std::string& src)
{
	return src;
}


template<typename T>
void Split(std::vector<T>& ret, const std::string &str, char delim, bool ignoreEmpty = true)
{
	if (str.empty()) {
		return;
	}
	ret.clear();

	size_t n = str.size();
	size_t s = 0;

	while (s <= n)
	{
		size_t i = str.find_first_of(delim, s);
		size_t len = 0;
		//T tmp;

		if (i == std::string::npos) {
			len = n - s;
		}
		else {
			len = i - s;
		}

		if (false == ignoreEmpty || 0 != len) {
			std::string tmp = str.substr(s, len);
			ret.push_back(std::move(exchange<T>(tmp)));
		}

		s += len + 1;
	}
}

template __declspec(dllexport) void Split(
  std::vector<std::string>& ret, const std::string &str, char delim, bool ignoreEmpty);
template __declspec(dllexport) void Split(
  std::vector<int>& ret, const std::string &str, char delim, bool ignoreEmpty);

}
