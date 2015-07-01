#define BOOST_TEST_MODULE RBM_ConvolutionalCDTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/ConvolutionalBinaryRBM.h>
#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include <shark/Unsupervised/RBM/analytics.h>
#include "../ObjectiveFunctions/TestObjectiveFunction.h"

#include <vector>
#include <fstream>
using namespace shark;
using namespace std;

//calculats the *exact* CD gradient and check that the CD gradient approaches it in the limit
BOOST_AUTO_TEST_SUITE (RBM_ConvolutionalCDTraining)

BOOST_AUTO_TEST_CASE( ConvolutionalCDTraining_Bars ){
	
	unsigned int trials = 1;
	unsigned int steps = 7001;
	unsigned int updateStep = 1000;
	
	BarsAndStripes problem(9);
	UnlabeledData<RealVector> data = problem.data();
	
	Rng::seed(42);
	
	ConvolutionalBinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4,5,3);
	
	for(unsigned int trial = 0; trial != trials; ++trial){
		Rng::seed(42+trial);
		RealVector params(rbm.numberOfParameters());
		for(std::size_t i = 0; i != params.size();++i){
			params(i) = Rng::uni(-0.1,0.1);
		}
		rbm.setParameterVector(params);
		ConvolutionalBinaryCD cd(&rbm);
		cd.setData(data);
		cd.setK(10);
		SteepestDescent optimizer;
		optimizer.setLearningRate(0.05);
		optimizer.setMomentum(0);
		optimizer.init(cd);
	
		double logLikelyHood = 0;
		for(std::size_t i = 0; i != steps; ++i){
			if(i % updateStep == 0){
				//std::cout<<partitionFunction(rbm);
				rbm.setParameterVector(optimizer.solution().point);
				logLikelyHood = negativeLogLikelihood(rbm,data);
				std::cout<<i<<" "<<logLikelyHood<<std::endl;
			}
			optimizer.step(cd);
		}
		BOOST_CHECK( logLikelyHood<200.0 );
	}
}


BOOST_AUTO_TEST_SUITE_END()
