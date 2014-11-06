#define BOOST_TEST_MODULE EvalSkipMissingFeaturesTestModule

#include "shark/Models/Kernels/EvalSkipMissingFeatures.h"
#include "shark/Models/Kernels/LinearKernel.h"

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>

namespace shark {

BOOST_AUTO_TEST_SUITE (Models_Kernels_EvalSkipMissingFeaturesTests)

BOOST_AUTO_TEST_CASE(TestEvalSkipMissingFeatures)
{
	// This test case is testing evaluating kernel function while skipping missing features

	const double tolerancePct = 0.001;

	// Test that calculation will take all features into consideration when missingness is set to boost::none
	LinearKernel<> kernel;
	RealVector inputA(5);
	inputA(0) = 1.0;
	inputA(1) = std::numeric_limits<double>::quiet_NaN();
	inputA(2) = 1.0;
	inputA(3) = 1.0;
	inputA(4) = 1.0;
	RealVector inputB(5);
	inputB(0) = 1.0;
	inputB(1) = 2.0;
	inputB(2) = std::numeric_limits<double>::quiet_NaN();
	inputB(3) = 4.0;
	inputB(4) = 5.0;

	// empty missingness will be ignored completed
	{
		const double actual = evalSkipMissingFeatures<RealVector>(kernel, inputA, inputB);
		BOOST_CHECK_CLOSE(actual, 10.0, tolerancePct);
	}

	{
		// No missing feature should be able to evaluated correctly
		RealVector missingness(5);
		missingness(0) = 1.0;
		missingness(1) = 1.0;
		missingness(2) = 1.0;
		missingness(3) = 1.0;
		missingness(4) = 1.0;
		const double actual1 = evalSkipMissingFeatures<RealVector>(kernel, inputA, inputB, missingness);
		BOOST_CHECK_CLOSE(actual1, 10.0, tolerancePct);

		// A missing feature should affect evaluation accordingly
		missingness(4) = std::numeric_limits<double>::quiet_NaN();
		const double actual2 = evalSkipMissingFeatures<RealVector>(kernel, inputA, inputB, missingness);
		BOOST_CHECK_CLOSE(actual2, 5.0, tolerancePct);
	}
}

// TODO: A interesting measurement:
// safety check comparing 1000 "normal" kernel evaluations to 1000 kernel evaluation where just the number of
// dimensions was doubled and the added dimensions filled with NaNs, and then see check the time penalty.
// Current implementation would not be a fast solution given those memory copies between vectors.

} // namespace shark {


BOOST_AUTO_TEST_SUITE_END()
