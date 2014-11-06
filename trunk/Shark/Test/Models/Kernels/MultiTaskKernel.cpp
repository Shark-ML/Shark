
#define BOOST_TEST_MODULE ML_MultiTaskKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/math/constants/constants.hpp>

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/MultiTaskKernel.h>


namespace shark {

// This unit test checks correctness of the
// GaussianTaskKernel class by comparing to
// a manually computed kernel value, and of
// the MultiTaskKernel wrapper class by
// checking that the product is correct.
BOOST_AUTO_TEST_SUITE (Models_Kernels_MultiTaskKernel)

BOOST_AUTO_TEST_CASE( MultiTaskKernel_Test )
{
	// define dummy data composed of an input and a task component
	RealVector x1(3); x1(0) = 1.0; x1(1) = 2.0; x1(2) = 3.0;
	RealVector x2(3); x2(0) = 0.0; x2(1) = 3.14159; x2(2) = 10.0 / 3.0;
	RealVector x3(3); x3(0) = 1.5; x3(1) = 2.5; x3(2) = 2.5;
	RealVector x4(3); x4(0) = 0.5; x4(1) = 3.0; x4(2) = 3.5;

	// composition of vector and task index
	MultiTaskSample<RealVector> v1(x1, 0);
	MultiTaskSample<RealVector> v2(x2, 1);
	MultiTaskSample<RealVector> v3(x3, 0);
	MultiTaskSample<RealVector> v4(x4, 1);
	std::vector<MultiTaskSample<RealVector> > vec;
	vec.push_back(v1);
	vec.push_back(v2);
	vec.push_back(v3);
	vec.push_back(v4);

	// build a data set object
	Data<MultiTaskSample<RealVector> > data = createDataFromRange(vec);
	
	// define a Gaussian kernel on inputs and a special task kernel on tasks
	const double gamma = 3.0;
	GaussianRbfKernel<RealVector> gauss(1.0);
	GaussianTaskKernel<RealVector> taskkernel(data, 2, gauss, gamma);
	MultiTaskKernel<RealVector> multitaskkernel(&gauss, &taskkernel);

	// compute kernel on the inputs
	RealMatrix g(data.numberOfElements(), data.numberOfElements());
	for (std::size_t i=0; i<data.numberOfElements(); i++)
	{
		for (std::size_t j=0; j<=i; j++)
		{
			double k = gauss(data.element(i).input, data.element(j).input);
			g(i, j) = g(j, i) = k;
		}
	}

	// target accuracy
	const double tolerance = 1e-14;

	// test diagonal entries of the Gaussian task kernel
	BOOST_CHECK_CLOSE(taskkernel(0, 0), 1.0, tolerance);
	BOOST_CHECK_CLOSE(taskkernel(1, 1), 1.0, tolerance);

	// test off-diagonal entries of the Gaussian task kernel
	const double k00 = (g(0, 0) + g(0, 2) + g(2, 0) + g(2, 2)) / 4.0;
	const double k01 = (g(0, 1) + g(0, 3) + g(1, 2) + g(2, 3)) / 4.0;
	const double k11 = (g(1, 1) + g(1, 3) + g(3, 1) + g(3, 3)) / 4.0;
	const double expected = std::exp(-gamma * (k00 - 2.0 * k01 + k11));
	BOOST_CHECK_CLOSE(taskkernel(0, 1), expected, tolerance);
	BOOST_CHECK_CLOSE(taskkernel(1, 0), expected, tolerance);

	// check the product property of the composed multi-task kernel
	BOOST_CHECK_CLOSE(g(0, 1) * expected, multitaskkernel.eval(v1, v2), tolerance);
}

} // namespace shark {

BOOST_AUTO_TEST_SUITE_END()
