#define BOOST_TEST_MODULE RBM_PCDTraining
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Unsupervised/RBM/BinaryRBM.h>
#include <shark/Unsupervised/RBM/analytics.h>

#include <shark/Unsupervised/RBM/Problems/BarsAndStripes.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_PCDTraining)

BOOST_AUTO_TEST_CASE( PCDTraining_Bars ){
	
	BarsAndStripes problem;
	UnlabeledData<RealVector> data = problem.data();
	
	random::globalRng.seed(0);
	
	BinaryRBM rbm(random::globalRng);
	rbm.setStructure(16,8);
	initRandomUniform(rbm,-0.1,0.1);
	BinaryPCD cd(&rbm);
	cd.setNumberOfSamples(32);
	cd.setBatchSize(16);
	cd.setData(data);
	
	
	SteepestDescent<> optimizer;
	optimizer.setLearningRate(0.05);
	optimizer.setMomentum(0);
	optimizer.init(cd);
	
	
	double logLikelyhood = 0;
	for(std::size_t i = 0; i != 5001; ++i){
		if(i % 5000 == 0){
			rbm.setParameterVector(optimizer.solution().point);
			logLikelyhood = negativeLogLikelihood(rbm,data);
			std::cout<<i<<" "<<logLikelyhood<<std::endl;
		}
		optimizer.step(cd);
	}
	
	BOOST_CHECK( logLikelyhood<200.0 );
}

BOOST_AUTO_TEST_SUITE_END()
