#define BOOST_TEST_MODULE MissingFeaturesKernelExpansionTestModule

#include "shark/Models/Kernels/LinearKernel.h"
#include "shark/Models/Kernels/MissingFeaturesKernelExpansion.h"
#include "../../Utils.h"

#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/assign/std/vector.hpp>

namespace shark {

BOOST_AUTO_TEST_SUITE (Models_Kernels_MissingFeaturesKernelExpansionTests)

BOOST_AUTO_TEST_CASE(TestMissingFeaturesKernelExpansion)
{
	// This test case is testing evaluating kernel function while skipping missing features

	const unsigned int featureSize = 4u;
	const unsigned int sampleSize = 5u;
	LinearKernel<> kernel;

	// Dataset
	std::vector<RealVector> input(sampleSize,RealVector(featureSize));
	std::vector<unsigned int> target(sampleSize);
	input[0](0) =  0.0; input[0](1) =  0.0; input[0](2) =  1.0; input[0](3) =  5.0; target[0] = 0;
	input[1](0) =  2.0; input[1](1) =  2.0; input[1](2) =  2.0; input[1](3) =  4.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) = -8.0; input[2](2) =  3.0; input[2](3) =  3.0; target[2] = 0;
	input[3](0) = -1.0; input[3](1) = -1.0; input[3](2) =  4.0; input[3](3) =  2.0; target[3] = 0;
	input[4](0) =  3.0; input[4](1) =  3.0; input[4](2) =  5.0; input[4](3) =  1.0; target[4] = 1;
	Data<RealVector> basis  = createDataFromRange(input);
	
	// The class under test
	MissingFeaturesKernelExpansion<RealVector> ke(&kernel, basis,false);

	// Scaling coefficients
	RealVector scalingCoefficients(sampleSize);
	scalingCoefficients(0) = 1.0;
	scalingCoefficients(1) = 0.2;
	scalingCoefficients(2) = 0.3;
	scalingCoefficients(3) = 0.4;
	scalingCoefficients(4) = 0.5;
	ke.setScalingCoefficients(scalingCoefficients);
	
	// Alphas
	RealVector alpha(sampleSize);
	alpha(0) = 1.0;
	alpha(1) = 0.2;
	alpha(2) = 0.3;
	alpha(3) = 0.4;
	alpha(4) = 0.5;
	ke.setParameterVector(alpha);
	ke.setClassifierNorm(10.0);
	
	// Do an evaluation and then verify
	RealVector pattern(featureSize);
	pattern(0) = 1.0;
	pattern(1) = 2.0;
	pattern(2) = std::numeric_limits<double>::quiet_NaN(); // missing feature
	pattern(3) = 4.0;
	RealVector output = ke(pattern);
	std::cout<<"d"<<std::endl;

	using namespace boost::assign;
	std::vector<double> expected;
	expected += 34.7851;
	BOOST_CHECK(test::verifyVectors(output, expected, 1e-2));
}

} // namespace shark {

BOOST_AUTO_TEST_SUITE_END()
