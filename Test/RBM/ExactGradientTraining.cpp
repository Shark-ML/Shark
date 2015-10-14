#define BOOST_TEST_MODULE RBM_ExactGradientTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/GradientApproximations/ExactGradient.h>
#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>

#include <vector>
#include <fstream>
using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (RBM_ExactGradientTraining)

BOOST_AUTO_TEST_CASE( ExactGradientTraining_Bars ){
	
	unsigned int trials = 1;
	unsigned int steps = 1001;
	unsigned int updateStep = 100;
	
	BarsAndStripes problem;
	UnlabeledData<RealVector> data = problem.data();
	
	
	
	for(unsigned int trial = 0; trial != trials; ++trial){
		Rng::seed(42+trial);
		BinaryRBM rbm(Rng::globalRng);
		rbm.setStructure(16,8);
		RealVector params(rbm.numberOfParameters());
		for(std::size_t i = 0; i != params.size();++i){
			params(i) = Rng::uni(-0.1,0.1);
		}
		rbm.setParameterVector(params);
		ExactGradient<BinaryRBM> gradient(&rbm);
		gradient.setData(data);
		SteepestDescent optimizer;
		optimizer.setLearningRate(0.2);
		optimizer.setMomentum(0);
		optimizer.init(gradient);
	
		double logLikelyHood = 0;
		for(std::size_t i = 0; i != steps; ++i){
			if(i % updateStep == 0){
				std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
			}
			optimizer.step(gradient);
		}
		BOOST_CHECK( logLikelyHood<200.0 );
	}
}

BOOST_AUTO_TEST_SUITE_END()
