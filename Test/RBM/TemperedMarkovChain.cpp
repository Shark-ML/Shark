#include <shark/Unsupervised/RBM/BinaryRBM.h>

#define BOOST_TEST_MODULE RBM_TemperedMarkovChain
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_SUITE (RBM_TemperedMarkovChain)

BOOST_AUTO_TEST_CASE( TemperedMarkovChain_Distribution )
{
	const std::size_t numTemperatures = 10;
	const std::size_t numSamples = 10000;
	
	double states[]={
		0,0,0,0,
		1,0,0,0,
		0,1,0,0,
		1,1,0,0,
		0,0,1,0,
		1,0,1,0,
		0,1,1,0,
		1,1,1,0,
		0,0,0,1,
		1,0,0,1,
		0,1,0,1,
		1,1,0,1,
		0,0,1,1,
		1,0,1,1,
		0,1,1,1,
		1,1,1,1,
	};
	RealMatrix stateMatrix = blas::adapt_matrix(16,4,states);
	
	//create rbm and pt object
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4);
	RealVector params(rbm.numberOfParameters());
	for(std::size_t i = 0; i != params.size();++i){
		params(i) = Rng::uni(-1,1);
	}
	rbm.setParameterVector(params);
	
	TemperedMarkovChain<GibbsOperator<BinaryRBM> > pt(&rbm);
	pt.setNumberOfTemperatures(numTemperatures);
	RealVector beta(numTemperatures);
	for(std::size_t i = 0; i != numTemperatures; ++i){
		double factor = numTemperatures - 1;
		beta(i) = 1.0 - i/factor;
		pt.setBeta(i,1.0 - i/factor);
	}
	pt.initializeChain(RealMatrix(numTemperatures,4,0));
	pt.step(1000);//burn in
	
	//evaluate distribution for all beta values
	RealMatrix pHidden(numTemperatures,16);
	RealMatrix pVisible(numTemperatures,16);
	for(std::size_t i = 0; i != numTemperatures; ++i){
		row(pHidden,i) =  exp(rbm.energy().logUnnormalizedProbabilityHidden(stateMatrix,blas::repeat(beta(i),16)));
		row(pVisible,i) =  exp(rbm.energy().logUnnormalizedProbabilityVisible(stateMatrix,blas::repeat(beta(i),16)));
		//normalize to 1
		row(pHidden,i) /= sum(row(pHidden,i));
		row(pVisible,i) /= sum(row(pVisible,i));
	}
	
	RealMatrix pHiddenHist(numTemperatures,16,0.0);
	RealMatrix pVisibleHist(numTemperatures,16,0.0);
	
	for(std::size_t s = 0; s != numSamples; ++s){
		pt.step(1);
		
		//get state number for every sampled state and add the sample to the histogram
		for(std::size_t t = 0; t != numTemperatures; ++t){
			std::size_t stateH = 0;
			std::size_t stateV = 0;
			for(std::size_t i = 0; i != 4; ++i){
				stateH += pt.samples().hidden.state(t,i) > 0? (1<<i):0; 
				stateV += pt.samples().visible.state(t,i) > 0? (1<<i):0; 
			}
			pHiddenHist(t,stateH)+=1.0/numSamples;
			pVisibleHist(t,stateV)+=1.0/numSamples;
		}
	}
	//calculate KL divergence between distributions
	for(std::size_t t = 0; t != numTemperatures; ++t){
		double KLV= sum(row(pVisible*log(pVisible/pVisibleHist),t));
		double KLH= sum(row(pHidden*log(pHidden/pHiddenHist),t));
		std::cout<<KLV <<" "<< KLH<<"\n";
		BOOST_CHECK_SMALL(KLV,0.01);
		BOOST_CHECK_SMALL(KLH,0.01);
	}
}

BOOST_AUTO_TEST_SUITE_END()
