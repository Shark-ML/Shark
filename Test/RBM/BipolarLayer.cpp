#include <shark/Unsupervised/RBM/Neuronlayers/BipolarLayer.h>
#include <shark/Core/Random.h>

#define BOOST_TEST_MODULE RBM_BipolarLayer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_BipolarLayer)

BOOST_AUTO_TEST_CASE( BipolarLayer_SufficientStatistics){
	BipolarLayer layer;
	BipolarLayer::SufficientStatistics statistics(10,3);
	layer.resize(3);
	RealMatrix input(10,3);
	RealMatrix testInput(10,3);
	random::globalRng.seed(42);
	
	for(std::size_t test = 0; test != 10000; ++test){
		double beta = random::uni(random::globalRng,0,1);
		for(std::size_t j = 0; j != 3; ++j){
			layer.bias()(j) = random::gauss(random::globalRng,0,10);
		}
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				input(i,j) = random::gauss(random::globalRng,0,10);
			}
			
			//calculate result
			row(testInput,i) = row(input,i) + layer.bias();
			row(testInput,i) *= beta;
		}
		layer.sufficientStatistics(input,statistics,blas::repeat(beta,10));
		
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 3; ++j){
				double sigmoid = 1.0/(1+std::exp(-2*testInput(i,j)));
				BOOST_CHECK_SMALL(sigmoid-statistics(i,j),1.e-15);
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( BipolarLayer_Parameters){
	BipolarLayer layer;
	layer.resize(20);
	RealVector parameters(20);
	random::globalRng.seed(42);
	
	for(std::size_t j = 0; j != 20; ++j){
		parameters(j) = random::gauss(random::globalRng,0,1);
	}
	
	layer.setParameterVector(parameters);
	
	for(std::size_t j = 0; j != 20; ++j){
		BOOST_CHECK_SMALL(layer.parameterVector()(j) - parameters(j),1.e-15);
	}
}

BOOST_AUTO_TEST_CASE( BipolarLayer_Sample){
	BipolarLayer layer;
	BipolarLayer::SufficientStatistics statistics(10,5);
	layer.resize(5);
	random::globalRng.seed(42);
	
	std::vector<double> alphas;
	alphas.push_back(0);
	alphas.push_back(1);
	for(std::size_t i = 0; i != 10; ++i){
		alphas.push_back(random::uni(random::globalRng,0,1));
	}
	
	const std::size_t numSamples = 100000;
	const double epsilon = 1.e-4;
	
	
	for(std::size_t a = 0; a != alphas.size(); ++a){
		double alpha = alphas[a];
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 5; ++j){
				statistics(i,j) = random::uni(random::globalRng,0.0,1.0);
			}
		}
		
		
		RealMatrix mean(10,5,0.0);
		RealMatrix samples(10,5,0.0);
		for(std::size_t i = 0; i != numSamples; ++i){
			layer.sample(statistics,samples,alpha,random::globalRng);
			mean+=samples;
		}
		mean/=numSamples;
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 5; ++j){
				BOOST_CHECK_SMALL(sqr(mean(i,j) - (2*statistics(i,j)-1)),epsilon);
			}
		}
	}
}

	

BOOST_AUTO_TEST_CASE( BipolarLayer_LogMarginalize){
	BipolarLayer layer;
	layer.resize(3);
	RealVector input(3);
	random::globalRng.seed(42);

	for(std::size_t j = 0; j != 3; ++j){
		layer.bias()(j) = -0.5*j-1;
	}
	
	for(std::size_t j = 0; j != 3; ++j){
		input(j) = -0.5*j-2;
	}	

	double testResult = std::log((std::exp(3.0)+std::exp(-3.0))*(std::exp(4.0)+std::exp(-4.0))*(std::exp(5.0)+std::exp(-5.0)) );
	double testResultHalvedBeta = std::log((std::exp(1.5)+std::exp(-1.5))*(std::exp(2.0)+std::exp(-2.0))*(std::exp(2.5)+std::exp(-2.5)));

	double result = layer.logMarginalize(input,1);
	double resultHalvedBeta = layer.logMarginalize(input,0.5);

	BOOST_CHECK_CLOSE(result , testResult , 0.01);
	BOOST_CHECK_CLOSE(resultHalvedBeta , testResultHalvedBeta , 0.01);
 }

BOOST_AUTO_TEST_SUITE_END()
