#include <shark/Unsupervised/RBM/Neuronlayers/TruncatedExponentialLayer.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Rng/TruncatedExponential.h>

#define BOOST_TEST_MODULE RBM_TruncatedExponentialLayer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( TruncatedExponentialLayer_SufficientStatistics){
	TruncatedExponentialLayer layer;
	TruncatedExponentialLayer::StatisticsBatch statistics(10,3);
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
			row(testInput,i) = row(input,i) + layer.bias();
			row(testInput,i) *= beta;
		}
		layer.sufficientStatistics(input,statistics,blas::repeat(beta,10));
		
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				BOOST_CHECK_SMALL(testInput(i,j)+statistics.lambda(i,j),1.e-10);
				BOOST_CHECK_SMALL(exp(testInput(i,j))-statistics.expMinusLambda(i,j),1.e-8);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( TruncatedExponentialLayer_Parameters){
	TruncatedExponentialLayer layer;
	layer.resize(20);
	RealVector parameters(20);
	Rng::seed(42);
	
	for(std::size_t j = 0; j != 20; ++j){
		parameters(j) = Rng::gauss(0,1);
	}
	
	layer.setParameterVector(parameters);
	
	for(std::size_t j = 0; j != 20; ++j){
		BOOST_CHECK_SMALL(layer.parameterVector()(j) - parameters(j),1.e-10);
		BOOST_CHECK_SMALL(layer.bias()(j) - parameters(j),1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( TruncatedExponentialLayer_Sample){
	TruncatedExponentialLayer layer;
	TruncatedExponentialLayer::StatisticsBatch statistics(10,5);
	layer.resize(5);
	Rng::seed(42);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			statistics.lambda(i,j) = Rng::uni(-1,1);
		}
	}
	statistics.expMinusLambda = exp(-statistics.lambda);
	
	const std::size_t numSamples = 100000;
	RealMatrix mean(10,5);
	mean.clear();
	for(std::size_t i = 0; i != numSamples; ++i){
		RealMatrix samples(10,5);
		layer.sample(statistics,samples,Rng::globalRng);
		mean+=samples;
	}
	mean/=numSamples;
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 5; ++j){
			double analyticMean = 1.0/statistics.lambda(i,j) -1.0/(std::exp(statistics.lambda(i,j))-1);
			BOOST_CHECK_CLOSE(mean(i,j) , analyticMean,1.);
		}
	}
}

BOOST_AUTO_TEST_CASE( TruncatedExponentialLayer_Marginalize){
	TruncatedExponentialLayer layer;
	layer.resize(3);
	RealVector input(3);
	Rng::seed(42);
	
	for(std::size_t j = 0; j != 3; ++j){
		layer.bias()(j) = -0.5*j-1;
	}
	for(std::size_t j = 0; j != 3; ++j){
		input(j) = -0.5*j-2;
	}
	
	
	double testResult = std::log(0.316738 * 0.245421 * 0.198652);
	double testResultHalvedBeta = std::log(0.517913 * 0.432332 * 0.367166);
	
	double result = layer.logMarginalize(input,1);
	double resultHalvedBeta = layer.logMarginalize(input,0.5);
	
	BOOST_CHECK_CLOSE(result , testResult , 0.01);
	BOOST_CHECK_CLOSE(resultHalvedBeta , testResultHalvedBeta , 0.01);
}

