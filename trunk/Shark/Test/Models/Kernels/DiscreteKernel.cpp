
#define BOOST_TEST_MODULE ML_DiscreteKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boost/math/constants/constants.hpp>

#include <shark/Models/Kernels/DiscreteKernel.h>
#include <cmath>

using namespace shark;


// This unit test checks correctness of the
// DiscreteKernel class.
BOOST_AUTO_TEST_SUITE (Models_Kernels_DiscreteKernel)

BOOST_AUTO_TEST_CASE( DiscreteKernel_Test )
{
	// define a positive definite, symmetric matrix
	RealMatrix mat(3, 3);
	mat(0, 0) = 2.0; mat(0, 1) = 1.0; mat(0, 2) = 0.5;
	mat(1, 0) = 1.0; mat(1, 1) = 2.0; mat(1, 2) = 1.5;
	mat(2, 0) = 0.5; mat(2, 1) = 1.5; mat(2, 2) = 1.25;

	// define the kernel
	DiscreteKernel k(mat);

	// target accuracy
	const double tolerance = 1e-14;

	// test kernel values and symmetry
	for (std::size_t i=0; i<mat.size1(); i++)
	{
		// diagonal
		BOOST_CHECK_SMALL(k(i, i) - mat(i, i), tolerance);

		// off-diagonal
		for (std::size_t j=0; j<i; j++)
		{
			BOOST_CHECK_SMALL(k(i, j) - mat(i, j), tolerance);
			BOOST_CHECK_SMALL(k(i, j) - k(j, i), tolerance);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
