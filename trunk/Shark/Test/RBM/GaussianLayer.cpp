#include <shark/Unsupervised/RBM/Neuronlayers/GaussianLayer.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE RBM_GaussianLayer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( GaussianLayer_SufficientStatistics){
	GaussianLayer layer;
	RealMatrix statistics(10,3);
	layer.resize(3);
	RealMatrix input(10,3);
	RealMatrix testInput(10,3);
	Rng::seed(42);
	
	for(std::size_t test = 0; test != 10000; ++test){
		double beta = Rng::uni(0,1);
		for(std::size_t j = 0; j != 3; ++j){
			layer.bias()(j) = Rng::gauss(0,10);
		}
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				input(i,j) = Rng::gauss(0,10);
			}
			
			//calculate result
			row(testInput,i) = row(input,i)*beta + layer.bias();
		}
		layer.sufficientStatistics(input,statistics,blas::repeat(beta,10));
		
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				BOOST_CHECK_SMALL(testInput(i,j)-statistics(i,j),1.e-15);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( GaussianLayer_Parameters){
	GaussianLayer layer;
	layer.resize(20);
	RealVector parameters(20);
	Rng::seed(42);
	
	for(std::size_t j = 0; j != 20; ++j){
		parameters(j) = Rng::gauss(0,1);
	}
	
	layer.setParameterVector(parameters);
	
	for(std::size_t j = 0; j != 20; ++j){
		BOOST_CHECK_SMALL(layer.parameterVector()(j) - parameters(j),1.e-15);
	}
}

BOOST_AUTO_TEST_CASE( GaussianLayer_Sample){
	GaussianLayer layer;
	GaussianLayer::StatisticsBatch statistics(3,5);
	layer.resize(5);
	Rng::seed(42);
	
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			statistics(i,j) = Rng::uni(0.0,1.0);
		}
	}
	

	const std::size_t numSamples = 5000000;
	RealMatrix mean(3,5);
	RealMatrix variance(3,5);
	mean.clear();
	variance.clear();
	RealMatrix samples(3,5);
	for(std::size_t s = 0; s != numSamples; ++s){
		layer.sample(statistics,samples,0.0,Rng::globalRng);
		mean +=samples;
		noalias(variance) += sqr(samples-statistics);
	}
	mean/=numSamples;
	variance/=numSamples;
	for(std::size_t i = 0; i != 3; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			BOOST_CHECK_SMALL(sqr(mean(i,j) - statistics(i,j)),1.e-5);
			BOOST_CHECK_SMALL(sqr(variance(i,j) - 1),1.e-2);
		}
	}
}

BOOST_AUTO_TEST_CASE( GaussianLayer_Marginalize){
	GaussianLayer layer;
	layer.resize(3);
	RealVector input(3);
	Rng::seed(42);
	
	//set distribution, sample input and calculate statistics
	for(std::size_t j = 0; j != 3; ++j){
		layer.bias()(j) = 0.5*j+1;
	}
	for(std::size_t j = 0; j != 3; ++j){
		input(j) = 0.5*j+2;
	}
	double result = layer.logMarginalize(input,1);
	double resultHalvedBeta = layer.logMarginalize(input,0.5);
	//result by wolfram alpha
	double testResult =  std::log(225.639 * 7472.15 * 672622);
	double testResultHalvedBeta = std::log(33.6331 * 193.545 * 1836.31);
	
	BOOST_CHECK_CLOSE(result, testResult, 0.01);
	BOOST_CHECK_CLOSE(resultHalvedBeta, testResultHalvedBeta, 0.01);
}
