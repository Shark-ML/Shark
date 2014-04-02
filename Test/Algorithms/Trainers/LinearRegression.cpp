#define BOOST_TEST_MODULE TRAINERS_LINEARREGRESSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LinearRegression.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/Rng/Uniform.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/LinAlg/rotations.h>

using namespace shark;

BOOST_AUTO_TEST_CASE( LinearRegression_TEST ){
	const size_t trainExamples = 60000;
	LinearRegression trainer;
	LinearModel<> model;
	RealMatrix matrix(2, 2);
	RealVector offset(2);
	matrix(0,0) = 3;
	matrix(1,1) = -5;
	matrix(0,1) = -2;
	matrix(1,0) = 7;
	offset(0) = 3;
	offset(1) = -6;
	model.setStructure(matrix, offset);

	// create datatset - the model output + Gaussian noise
	RealMatrix covariance(2, 2);
	covariance(0,0) = 1;
	covariance(0,1) = 0;
	covariance(1,0) = 0;
	covariance(1,1) = 1;
	MultiVariateNormalDistribution noise(covariance);

	Uniform<> uniform(Rng::globalRng,-3.0, 3.0);

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<RealVector> trainTarget(trainExamples,RealVector(2));
	std::vector<RealVector> testTarget(trainExamples,RealVector(2));
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = uniform();
		input[i](1) = uniform();
		testTarget[i] =  model(input[i]);
		trainTarget[i] = noise().first + testTarget[i];
	}

	// let the model forget...
 	matrix.clear();
 	offset.clear();
 	model.setStructure(matrix,offset);

	// train the model, overwriting its parameters
	RegressionDataset trainset = createLabeledDataFromRange(input, trainTarget);
	trainer.train(model, trainset);

	// evaluate using the ErrorFunction
	RegressionDataset testset = createLabeledDataFromRange(input, testTarget);
	SquaredLoss<> loss;
	double error=loss(testset.labels(),model(testset.inputs()));
	BOOST_CHECK_SMALL(error, 1e-4);
}
