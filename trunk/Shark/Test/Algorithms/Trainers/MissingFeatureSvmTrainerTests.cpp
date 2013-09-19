#define BOOST_TEST_MODULE MissingFeaturesSvmTrainerTestModule

#include <boost/assign/list_of.hpp>
#include <boost/assign/std/vector.hpp>
#include <boost/format.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test.hpp>

// In this test case, we want to count the kernel lookups
#define SHARK_COUNT_KERNEL_LOOKUPS

#include <shark/Algorithms/Trainers/MissingFeatureSvmTrainer.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/Csv.h>
#include <shark/LinAlg/Base.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/TestHelpers/Utils.h>

namespace shark {

/// Test fixture
class SvmWafFixture
{
public:

	SvmWafFixture()
	{
		//~ string2data<RealVector, unsigned int>(m_labeledData, m_dataInString, LAST_COLUMN, true, false, NULL);
		//~ string2data<RealVector, unsigned int>(m_labeledData2, m_dataInString2, LAST_COLUMN, true, false, NULL);
		csvStringToData(m_labeledData,m_dataInString,LAST_COLUMN);
		csvStringToData(m_labeledData2,m_dataInString2,LAST_COLUMN);
	}

	static const std::string m_dataInString;
	static const std::string m_dataInString2;
	LabeledData<RealVector, unsigned int> m_labeledData;
	LabeledData<RealVector, unsigned int> m_labeledData2;
};

const std::string SvmWafFixture::m_dataInString = "\
?,180,12,0\n\
5.92,190,11,0\n\
5.58,170,12,0\n\
5.92,165,10,0\n\
5,100,6,1\n\
5.5,150,8,1\n\
5.42,130,7,1\n\
5.75,150,9,1\r";

/// The difference is adding additional one column of '?' to @a m_dataInString
const std::string SvmWafFixture::m_dataInString2 = "\
?,180,12,?,0\n\
5.92,190,11,?,0\n\
5.58,170,12,?,0\n\
5.92,165,10,?,0\n\
5,100,6,?,1\n\
5.5,150,8,?,1\n\
5.42,130,7,?,1\n\
5.75,150,9,?,1\r";

BOOST_FIXTURE_TEST_SUITE(SvmWafFixtureTests, SvmWafFixture)

BOOST_AUTO_TEST_CASE(NoMissingFeatures)
{
	// Test that out trainer works fine with normal feature set without Missing features
	// In this case, we should get the same results as class CSvmTrainer

	// simple 5-point dataset
	std::vector<RealVector> input(5);
	std::vector<unsigned int> target(5);
	for (std::size_t i=0; i<5; i++) input[i].resize(2);
	input[0](0) =  0.0; input[0](1) =  0.0; target[0] = 0;
	input[1](0) =  2.0; input[1](1) =  2.0; target[1] = 1;
	input[2](0) = -1.0; input[2](1) = -8.0; target[2] = 0;
	input[3](0) = -1.0; input[3](1) = -1.0; target[3] = 0;
	input[4](0) =  3.0; input[4](1) =  3.0; target[4] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	// Soft-margin training with linear kernel
	RealVector seenParam1;
	{
		LinearKernel<> kernel;
		MissingFeaturesKernelExpansion<RealVector> svm;
		MissingFeatureSvmTrainer<RealVector, double> trainer(&kernel, 0.1, true);
		trainer.setMaxIterations(1);
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.train(svm, dataset);
		seenParam1 = svm.parameterVector();

		// Test against analytically known solution
		// param 0-4 is alpha, param 5 is bias(offset)
		RealVector expected(6);
		expected(0) = -0.1;
		expected(1) = 0.1;
		expected(2) = 0.0;
		expected(3) = -0.0125;
		expected(4) = 0.0125;
		expected(5) = -0.5;
		BOOST_CHECK(test::verifyVectors(seenParam1, expected));
	}

	// The result should also be the same as results trained by CSvmTrainer
	{
		LinearKernel<> kernel;
		KernelClassifier<RealVector> svm;
		CSvmTrainer<RealVector> trainer(&kernel, 0.1, true);
		trainer.stoppingCondition().minAccuracy = 1e-8;
		trainer.sparsify() = false;
		trainer.train(svm, dataset);
		RealVector seenParam2 = svm.parameterVector();

		// seenParam1 and seenParam2 should be the same
		BOOST_CHECK(test::verifyVectors(seenParam1, seenParam2));
	}
}

BOOST_AUTO_TEST_CASE(MissingFeatures)
{
	RealVector pattern1(3);
	pattern1(0) = 5.85;
	pattern1(1) = std::numeric_limits<double>::quiet_NaN();
	pattern1(2) = 8.5;

	RealVector pattern2(4);
	pattern2(0) = 5.85;
	pattern2(1) = std::numeric_limits<double>::quiet_NaN();
	pattern2(2) = 8.5;
	pattern2(3) = 170.5;

	RealVector param1;
	RealVector output1;
	{
		LinearKernel<> kernel;
		MissingFeaturesKernelExpansion<RealVector> svm1;
		MissingFeatureSvmTrainer<RealVector, double> trainer(&kernel, 0.1, true);
		trainer.setMaxIterations(4);
		trainer.stoppingCondition().minAccuracy = 1e-8;

		// Train with normal data
		trainer.train(svm1, m_labeledData);
		param1 = svm1.parameterVector();
		output1 = svm1(pattern1);
	}

	RealVector param2;
	RealVector output2;
	{
		// Train with data with missing features
		LinearKernel<> kernel;
		MissingFeaturesKernelExpansion<RealVector> svm2;
		MissingFeatureSvmTrainer<RealVector, double> trainer(&kernel, 0.1, true);
		trainer.train(svm2, m_labeledData2);
		trainer.setMaxIterations(4);
		trainer.stoppingCondition().minAccuracy = 1e-8;
		param2 = svm2.parameterVector();

		// Try to make a prediction
		output2 = svm2(pattern2);
	}

	// Should be the same
	BOOST_CHECK(test::verifyVectors(param1, param2));
	BOOST_CHECK(test::verifyVectors(output1, output2));
}

BOOST_AUTO_TEST_SUITE_END()

} // namespace shark {
