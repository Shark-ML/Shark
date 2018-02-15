#include <shark/ObjectiveFunctions/NegativeLogLikelihood.h>
#include <shark/Models/RBFLayer.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>

#define BOOST_TEST_MODULE ObjFunct_NegativeLogLikelihood
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "TestObjectiveFunction.h"
using namespace shark;

struct Fixture {

	Fixture(): covariance(10,10,0.0),offset(10){
		for(std::size_t i = 0; i != 10; ++i){
			covariance(i,i) = 5;
			offset(i) = i;
		}
		NormalDistributedPoints problem(covariance,offset);
		data = problem.generateDataset(6000,100);
	}
	RealMatrix covariance;
	RealVector offset;
	UnlabeledData<RealVector> data;
};

BOOST_FIXTURE_TEST_SUITE (ObjectiveFunctions_NegativeLogLikelihood, Fixture)


BOOST_AUTO_TEST_CASE( ObjFunct_NegativeLogLikelihood_Derivative ){
	RBFLayer model(10,1);
	NegativeLogLikelihood function(data,&model);
	
	std::size_t numTrials = 1000;
	RealVector point(model.numberOfParameters());
	for(std::size_t t = 0; t != numTrials; ++t){
		for(std::size_t i = 0; i != model.numberOfParameters()-1; ++i){
			point(i) = random::gauss(random::globalRng, 0,1);
		}
		point(model.numberOfParameters()-1) = random::uni(random::globalRng, -5,-3);
		testDerivative(function,point,1.e-5,0,0.01);
	}
}

BOOST_AUTO_TEST_CASE( ObjFunct_NegativeLogLikelihood_Optimize ){
	RBFLayer model(10,1);
	RealVector point(model.numberOfParameters());
	for(std::size_t i = 0; i != model.numberOfParameters()-1; ++i){
		point(i) = random::gauss(random::globalRng, 0,1);
	}
	model.setParameterVector(point);
	
	NegativeLogLikelihood function(data,&model);
	Rprop<> optimizer;
	optimizer.init(function);
	
	for(std::size_t i = 0; i != 100; ++i){
		optimizer.step(function);
	}
	model.setParameterVector(optimizer.solution().point);
	BOOST_CHECK_SMALL(norm_inf(row(model.centers(),0)-offset), 0.1);
	
	std::cout<<model.centers();
}

BOOST_AUTO_TEST_SUITE_END()
