#ifndef SHARK_TEST_HELPERS_UTILS_H
#define SHARK_TEST_HELPERS_UTILS_H

#include <boost/format.hpp>
#include <boost/test/test_tools.hpp>

namespace shark { namespace test {

/// Verify whether two Vector's are equal
template<typename VectorType, typename VectorType2>
boost::test_tools::predicate_result verifyVectors(
	const VectorType& seen,
	const VectorType2& expected,
	const double tolerance = 1e-10)
{
	boost::test_tools::predicate_result res(true);
	if (seen.size() != expected.size())
	{
		res = false;
		res.message() << boost::format("Size mismatch: seen:%1%, expected:%2%") % seen.size() % expected.size();
		return res;
	}
	for (std::size_t i = 0; i < seen.size(); ++i)
	{
		if (std::abs(seen(i) - expected[i]) > tolerance)
		{
			res = false;
			res.message() << boost::format("index:%1%, seen:%2%, expected:%3%") % i % seen(i) % expected[i];
			break;
		}
	}

	return res;
}

}} // namespace shark { namespace test {

#endif // SHARK_TEST_HELPERS_UTILS_H
