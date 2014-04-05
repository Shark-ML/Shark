#define BOOST_TEST_MODULE RBM_ParallelTemperingTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/analytics.h>

#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include <fstream>
using namespace shark;

BOOST_AUTO_TEST_CASE( ParallelTemperingTraining_Bars ){
	
	unsigned int trials = 1;
	unsigned int steps = 3001;
	unsigned int updateStep = 1000;
	std::size_t numHidden = 8;
	std::size_t numTemperatures = 5;
	double learningRate = 0.1;
	
	BarsAndStripes problem(8);
	UnlabeledData<RealVector> data = problem.data();
	
	for(unsigned int trial = 0; trial != trials; ++trial){
		BinaryRBM rbm(Rng::globalRng);
		rbm.setStructure(16,numHidden);
		
		Rng::seed(42+trial);
		RealVector params(rbm.numberOfParameters());
		for(std::size_t i = 0; i != params.size();++i){
			params(i) = Rng::uni(-0.1,0.1);
		}
		rbm.setParameterVector(params);
		BinaryParallelTempering cd(&rbm);
		cd.chain().setNumberOfTemperatures(numTemperatures);
		for(std::size_t i = 0; i != numTemperatures; ++i){
			double factor = numTemperatures - 1;
			cd.chain().setBeta(i,1.0 - i/factor);
		}
		cd.numBatches()=2;
		cd.setData(data);

		SteepestDescent optimizer;
		optimizer.setLearningRate(learningRate);
		optimizer.setMomentum(0);
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
		BOOST_CHECK( logLikelyHood<200.0 );
	}
}
