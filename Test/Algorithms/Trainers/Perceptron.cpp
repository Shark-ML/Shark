#define BOOST_TEST_MODULE ML_PERCEPTRON
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/Perceptron.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Perceptron)

BOOST_AUTO_TEST_CASE( PERCEPTRON ){
	DenseLinearKernel kernel;
	KernelClassifier<RealVector> model;
	Perceptron<RealVector> trainer(&kernel);

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
	ZeroOneLoss<unsigned int> loss;
	BOOST_CHECK_SMALL(loss(dataset.labels(),model(dataset.inputs())),1.e-8);
}


BOOST_AUTO_TEST_SUITE_END()
