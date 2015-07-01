#define BOOST_TEST_MODULE RBM_ConvolutionalPTTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/ConvolutionalBinaryRBM.h>
#include <shark/Unsupervised/RBM/analytics.h>

#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include <fstream>
using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_ConvolutionalPTTraining)

BOOST_AUTO_TEST_CASE( ConvolutionalPTTraining_Bars ){
	
	unsigned int trials = 1;
	unsigned int steps = 7001;
	unsigned int updateStep = 1000;
	std::size_t numTemperatures = 10;
	double learningRate = 0.05;
	
	BarsAndStripes problem(8);
	UnlabeledData<RealVector> data = problem.data();
	
	for(unsigned int trial = 0; trial != trials; ++trial){
		ConvolutionalBinaryRBM rbm(Rng::globalRng);
		rbm.setStructure(4,4,5,3);
		
		Rng::seed(42+trial);
		RealVector params(rbm.numberOfParameters());
		for(std::size_t i = 0; i != params.size();++i){
			params(i) = Rng::uni(-0.1,0.1);
		}
		rbm.setParameterVector(params);
		ConvolutionalBinaryParallelTempering cd(&rbm);
		cd.chain().setUniformTemperatureSpacing(numTemperatures);
		cd.setData(data);

		SteepestDescent optimizer;
		optimizer.setLearningRate(learningRate);
		optimizer.init(cd);
	
		double logLikelyHood = 0;
		for(std::size_t i = 0; i != steps; ++i){
			if(i % updateStep == 0){
				rbm.setParameterVector(optimizer.solution().point);
				logLikelyHood = negativeLogLikelihood(rbm,data);
				std::cout<<i<<" "<<logLikelyHood<<std::endl;
			}
			optimizer.step(cd);
		}
		BOOST_CHECK( logLikelyHood<140.0 );
	}
}

BOOST_AUTO_TEST_SUITE_END()
