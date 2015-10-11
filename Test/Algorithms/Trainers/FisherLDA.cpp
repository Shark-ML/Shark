#define BOOST_TEST_MODULE ML_FISHER_LDA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/FisherLDA.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_FisherLDA)

BOOST_AUTO_TEST_CASE( FISHER_LDA_TEST ){
	const size_t trainExamples = 200000;
	FisherLDA trainer;
	trainer.setWhitening(false);
	LinearModel<> model(3, 3, true);

	// create datatsets - three normal distributions
	// [TG] why not use DataDistribution for this?

	RealMatrix covariance(3,3);
	covariance.clear();
	covariance(0,0)=1;
	covariance(1,1)=1;
	covariance(2,2)=1;

	RealVector mean[] = {RealVector(3), RealVector(3), RealVector(3)};
	mean[0](0) = 20;
	mean[0](1) = 0;
	mean[0](2) = 0;

	mean[1](0) = -10;
	mean[1](1) = 0;
	mean[1](2) = 20;

	mean[2](0) = -10;
	mean[2](1) = 0;
	mean[2](2) = -20;
	MultiVariateNormalDistribution dist(covariance);

	RealVector result[] = {RealVector(3), RealVector(3)};
	result[0].clear();
	result[0](2) = 1;
	result[1].clear();
	result[1](0) = 1;


	std::vector<RealVector> input(trainExamples, RealVector(3));
	std::vector<unsigned int> target(trainExamples);

	for(size_t i=0;i!=trainExamples;++i) {
		//create sample
		target[i] = i % 3;
		input[i] = dist().first + mean[target[i]];
	}
	//statisticalBayesRisk/=trainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	// test the direction
	for(size_t i = 0; i != 2; ++i){
		RealVector curRow = row(model.matrix(), i);
		std::cout << curRow << std::endl;
		double error = std::min(norm_sqr(curRow - result[i]), norm_sqr(curRow + result[i]));
		BOOST_CHECK_SMALL(error, 1e-4);		// [TG] 10e-4 ???
	}
}

BOOST_AUTO_TEST_SUITE_END()
