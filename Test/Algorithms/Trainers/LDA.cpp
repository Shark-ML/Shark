#define BOOST_TEST_MODULE TRAINERS_LDA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <shark/LinAlg/solveSystem.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
using namespace shark;

BOOST_AUTO_TEST_CASE( LDA_TEST_TWOCLASS ){
	const size_t trainExamples = 12000;
	LDA trainer;
	LinearClassifier<> model;


	//create datatsets - two overlapping normal distributions
	MultiVariateNormalDistribution dist;

	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);


	RealVector mean[]={RealVector(2),RealVector(2)};
	mean[0](0)=0;
	mean[0](1)=0;
	mean[1](0)=6;
	mean[1](1)=6;
	
	double prior[2]={std::log(1.0/3.0),std::log(2.0/3.0)};
	
	dist.setCovarianceMatrix(covariance);


	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<unsigned int> target(trainExamples);

	double statisticalBayesRisk=0;
	for(size_t i=0;i!=trainExamples;++i){
		//create samples. class 1 has double as many elements as class 0
		target[i]=(i%3 == 0);
		input[i]=dist().first+mean[target[i]];
		//calculate bayes Target - the best fit to the distributions
		RealVector diff=input[i]-mean[0];
		double dist0=inner_prod(diff,prod(inverse,diff))+prior[0];
		diff=input[i]-mean[1];
		double dist1=inner_prod(diff,prod(inverse,diff))+prior[1];
		unsigned int bayesTarget = dist0>dist1;
		statisticalBayesRisk+= bayesTarget != target[i];
	}
	statisticalBayesRisk/=trainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	
	Data<unsigned int> results = model(dataset.inputs());
	
	ZeroOneLoss<> loss;
	double classificatorRisk = loss.eval(dataset.labels(),results);

	std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,0.01);

}

BOOST_AUTO_TEST_CASE( LDA_TEST_TWOCLASS_SINGULAR ){
	const size_t trainExamples = 12000;
	LDA trainer;
	LinearClassifier<> model;


	//create datatsets - two overlapping normal distributions
	//same as in the previous test aside from that we add a third
	//variable which is allways 0
	MultiVariateNormalDistribution dist;

	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);


	RealVector mean[]={RealVector(2),RealVector(2)};
	mean[0](0)=0;
	mean[0](1)=0;
	mean[1](0)=6;
	mean[1](1)=6;
	dist.setCovarianceMatrix(covariance);
	
	double prior[2]={std::log(2.0/3.0),std::log(1.0/3.0)};


	std::vector<RealVector> input(trainExamples,RealVector(3));
	std::vector<unsigned int> target(trainExamples);

	double statisticalBayesRisk=0;
	for(size_t i=0;i!=trainExamples;++i){
		//create sample
		target[i]= (i%3 != 0);
		RealVector vec = dist().first+mean[target[i]];
		//calculate bayes Target - the best fit to the distributions
		RealVector diff=vec-mean[0];
		double dist0=inner_prod(diff,prod(inverse,diff)) + prior[0];
		diff=vec-mean[1];
		double dist1=inner_prod(diff,prod(inverse,diff)) + prior[1];
		unsigned int bayesTarget = dist0>dist1;
		statisticalBayesRisk+= bayesTarget != target[i];
		init(input[i])<<vec,0;//add third zero
	}
	statisticalBayesRisk/=trainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	
	Data<unsigned int> results = model(dataset.inputs());
	
	ZeroOneLoss<> loss;
	double classificatorRisk = loss.eval(dataset.labels(),results);

	std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,0.01);

}

BOOST_AUTO_TEST_CASE( LDA_TEST_MULTICLASS ){
	const size_t trainExamples = 200000;
	const unsigned int classes = 10;
	LDA trainer;
	LinearClassifier<> model;
	Rng::seed(44);


	//create datatsets - overlapping normal distributions
	MultiVariateNormalDistribution dist;

	RealMatrix covariance(2,2);
	covariance(0,0)=16;
	covariance(0,1)=8;
	covariance(1,0)=8;
	covariance(1,1)=16;
	RealMatrix inverse(2,2,0.0);
	inverse(0,0) = inverse(1,1) = 1.0;
	blas::solveSymmSystemInPlace<blas::SolveAXB>(covariance,inverse);
	
	std::vector<RealVector> mean(classes,RealVector(2));
	for(unsigned int c = 0; c != classes; ++c){
		for(std::size_t j = 0; j != 2; ++j){
			mean[c](j) = Rng::gauss(0,30);
		}
	}
	dist.setCovarianceMatrix(covariance);


	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<unsigned int> target(trainExamples);

	double statisticalBayesRisk=0;
	for(size_t i=0;i!=trainExamples;++i){
		//create sample
		target[i]=i%classes;
		input[i]=dist().first+mean[target[i]];
		//calculate bayes Target - the best fit to the distributions
		unsigned int bayesTarget = 0;
		double minDist = 1.e30;
		for(unsigned int c = 0; c != classes; ++c){
			RealVector diff=input[i]-mean[c];
			double dist=inner_prod(diff,prod(inverse,diff));
			if(dist<minDist){
				minDist=dist;
				bayesTarget=c;
			}
		}
		statisticalBayesRisk+= bayesTarget != target[i];
	}
	statisticalBayesRisk/=trainExamples;

	ClassificationDataset dataset = createLabeledDataFromRange(input,target);

	trainer.train(model, dataset);

	//double classificatorRisk=0;
	Data<unsigned int> results = model(dataset.inputs());
	
	ZeroOneLoss<> loss;
	double classificatorRisk = loss.eval(dataset.labels(),results);
	
	std::cout<<"statistical bayes Risk: "<<statisticalBayesRisk<<std::endl;
	std::cout<<"classificator Risk: "<<classificatorRisk<<std::endl;
	BOOST_CHECK_SMALL(classificatorRisk-statisticalBayesRisk,10e-2);

}
