#define BOOST_TEST_MODULE TRAINERS_LOGISTIC_REGRESSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LogisticRegression.h>
#include <shark/Data/DataDistribution.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Logistic_Regression)

BOOST_AUTO_TEST_CASE( LogReg_Binary_NoLone){
	PamiToy problem;
	ClassificationDataset const& dataset = problem.generateDataset(50);
	
	LogisticRegression<> trainer;
	BOOST_CHECK_EQUAL(trainer.lambda1(),0.0);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.0);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(0),0.0);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(1),0.0);
	BOOST_CHECK_EQUAL(trainer.parameterVector().size(),2);
	BOOST_CHECK_EQUAL(trainer.numberOfParameters(),2);
	trainer.setLambda2(0.1);
	BOOST_CHECK_EQUAL(trainer.lambda1(),0.0);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(0),0.0);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(1),0.1);
	BOOST_CHECK_EQUAL(trainer.accuracy(), 1.e-8);
	LogisticRegression<>::ModelType model;
	trainer.train(model,dataset);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),11);
	
	//check gradient
	{
		CrossEntropy<unsigned int, RealVector> loss;
		ErrorFunction<> error(dataset, &model.decisionFunction(),&loss);
		RealVector derivative;
		error.evalDerivative(model.parameterVector(),derivative);
		RealVector penalty = 0.1 * model.parameterVector();
		penalty(10) = 0;
		BOOST_CHECK_SMALL(norm_inf(derivative+penalty),1.e-8);
	}
}

BOOST_AUTO_TEST_CASE( LogReg_Binary_NoBias_NoLone){
	PamiToy problem;
	ClassificationDataset const& dataset = problem.generateDataset(50);
	
	LogisticRegression<> trainer(0,0.1,false);
	BOOST_CHECK_EQUAL(trainer.lambda1(),0.0);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.1);
	BOOST_CHECK_EQUAL(trainer.accuracy(), 1.e-8);
	LogisticRegression<>::ModelType model;
	trainer.train(model,dataset);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),10);
	
	//check gradient
	{
		CrossEntropy<unsigned int, RealVector> loss;
		ErrorFunction<> error(dataset, &model.decisionFunction(),&loss);
		RealVector derivative;
		error.evalDerivative(model.parameterVector(),derivative);
		RealVector penalty = 0.1 * model.parameterVector();
		BOOST_CHECK_SMALL(norm_inf(derivative+penalty),1.e-8);
	}
}

BOOST_AUTO_TEST_CASE( LogReg_Binary_Lone){
	random::globalRng.seed(42);
	PamiToy problem;
	ClassificationDataset const& dataset = problem.generateDataset(50);
	
	double lambda1 = 0.1;
	LogisticRegression<> trainer(lambda1,0.1);
	BOOST_CHECK_EQUAL(trainer.lambda1(),lambda1);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(0),lambda1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(1),0.1);
	BOOST_CHECK_EQUAL(trainer.accuracy(), 1.e-8);
	LogisticRegression<>::ModelType model;
	trainer.train(model,dataset);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),11);
	
	//check gradient
	{
		RealVector point = model.parameterVector();
		CrossEntropy<unsigned int, RealVector> loss;
		ErrorFunction<> error(dataset, &model.decisionFunction(),&loss);
		RealVector derivative;
		error.evalDerivative(point,derivative);
		RealVector penalty = 0.1 * point;
		penalty(10) = 0;
		derivative += penalty;
		
		//create subgradient derivative
		for(std::size_t i = 0; i != 10; ++i){
			if(std::abs(point(i)) > 1.e-13)
				derivative(i) += lambda1 * (point(i) > 0? 1.0 : -1.0);
			else//otherwise choose the proper subgradient that makes the resulting gradient smallest
				derivative(i) -= std::min(lambda1,std::abs(derivative(i))) * (derivative(i) > 0? 1.0 : -1.0);
		}
		
		std::cout<<point<<std::endl;
		std::cout<<derivative<<std::endl;
		
		BOOST_CHECK_SMALL(norm_inf(derivative),1.e-8);
	}
}

BOOST_AUTO_TEST_CASE( LogReg_Binary_NoBias_Lone){
	random::globalRng.seed(42);
	PamiToy problem;
	ClassificationDataset const& dataset = problem.generateDataset(50);
	
	double lambda1 = 0.1;
	LogisticRegression<> trainer(lambda1,0.1,false);
	BOOST_CHECK_EQUAL(trainer.lambda1(),lambda1);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(0),lambda1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(1),0.1);
	BOOST_CHECK_EQUAL(trainer.accuracy(), 1.e-8);
	LogisticRegression<>::ModelType model;
	trainer.train(model,dataset);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),10);
	
	//check gradient
	{
		RealVector point = model.parameterVector();
		CrossEntropy<unsigned int, RealVector> loss;
		ErrorFunction<> error(dataset, &model.decisionFunction(),&loss);
		RealVector derivative;
		error.evalDerivative(point,derivative);
		RealVector penalty = 0.1 * point;
		derivative += penalty;
		
		//create subgradient derivative
		for(std::size_t i = 0; i != 10; ++i){
			if(std::abs(point(i)) > 1.e-13)
				derivative(i) += lambda1 * (point(i) > 0? 1.0 : -1.0);
			else//otherwise choose the proper subgradient that makes the resulting gradient smallest
				derivative(i) -= std::min(lambda1,std::abs(derivative(i))) * (derivative(i) > 0? 1.0 : -1.0);
		}
		
		std::cout<<point<<std::endl;
		std::cout<<derivative<<std::endl;
		
		BOOST_CHECK_SMALL(norm_inf(derivative),1.e-8);
	}
}

BOOST_AUTO_TEST_CASE( LogReg_Binary_Lone_Weighted){
	random::globalRng.seed(42);
	PamiToy problem;
	ClassificationDataset const& baseDataset = problem.generateDataset(50);
	WeightedLabeledData<RealVector, unsigned int> dataset(baseDataset, 1.0);
	for(double& weight: dataset.weights().elements()){
		weight = random::uni(random::globalRng, 0.1,2);
	}
	double lambda1 = 0.1;
	LogisticRegression<> trainer(lambda1,0.1);
	BOOST_CHECK_EQUAL(trainer.lambda1(),lambda1);
	BOOST_CHECK_EQUAL(trainer.lambda2(),0.1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(0),lambda1);
	BOOST_CHECK_EQUAL(trainer.parameterVector()(1),0.1);
	BOOST_CHECK_EQUAL(trainer.accuracy(), 1.e-8);
	LogisticRegression<>::ModelType model;
	trainer.train(model,dataset);
	
	BOOST_CHECK_EQUAL(model.numberOfParameters(),11);
	
	//check gradient
	{
		RealVector point = model.parameterVector();
		CrossEntropy<unsigned int, RealVector> loss;
		ErrorFunction<> error(dataset, &model.decisionFunction(),&loss);
		RealVector derivative;
		error.evalDerivative(point,derivative);
		RealVector penalty = 0.1 * point;
		penalty(10) = 0;
		derivative += penalty;
		
		//create subgradient derivative
		for(std::size_t i = 0; i != 10; ++i){
			if(std::abs(point(i)) > 1.e-13)
				derivative(i) += lambda1 * (point(i) > 0? 1.0 : -1.0);
			else//otherwise choose the proper subgradient that makes the resulting gradient smallest
				derivative(i) -= std::min(lambda1,std::abs(derivative(i))) * (derivative(i) > 0? 1.0 : -1.0);
		}
		
		std::cout<<point<<std::endl;
		std::cout<<derivative<<std::endl;
		
		BOOST_CHECK_SMALL(norm_inf(derivative),1.e-8);
	}
}


BOOST_AUTO_TEST_SUITE_END()
