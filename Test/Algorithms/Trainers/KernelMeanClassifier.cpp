#define BOOST_TEST_MODULE ML_PERCEPTRON
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/KernelMeanClassifier.h>
#include <shark/Models/Kernels/LinearKernel.h>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_KernelMeanClassifier)

BOOST_AUTO_TEST_CASE( KERNEL_MEAN_CLASSIFIER ) {
	
	DenseLinearKernel kernel;
	KernelMeanClassifier<RealVector> trainer(&kernel);
	KernelClassifier<RealVector> model;

	std::vector<RealVector> input(6,RealVector(2));
	input[0](0)=1;
	input[0](1)=3;
	input[1](0)=-1;
	input[1](1)=3;
	input[2](0)=1;
	input[2](1)=0;
	input[3](0)=-1;
	input[3](1)=0;
	input[4](0)=1;
	input[4](1)=-3;
	input[5](0)=-1;
	input[5](1)=-3;
	std::vector<unsigned int> target(6);
	target[0]=0;
	target[1]=1;
	target[2]=0;
	target[3]=1;
	target[4]=0;
	target[5]=1;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	for(size_t i = 0; i != 6; ++i){
		RealVector result = model.decisionFunction()(input[i]);
		BOOST_CHECK_EQUAL(result.size(),1u);
		unsigned int label = result(0)>0;
		BOOST_CHECK_EQUAL(target[i],label);
	}

}


BOOST_AUTO_TEST_SUITE_END()
