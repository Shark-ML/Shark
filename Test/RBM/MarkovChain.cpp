#include <shark/Unsupervised/RBM/BinaryRBM.h>

#define BOOST_TEST_MODULE RBM_MarkovChain
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_CASE( MarkovChain_Distribution )
{
	const std::size_t batchSize = 16;
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
	
	//create rbm and chain
	BinaryRBM rbm(Rng::globalRng);
	rbm.setStructure(4,4);
	RealVector params(rbm.numberOfParameters());
	for(std::size_t i = 0; i != params.size();++i){
		params(i) = Rng::uni(-1,1);
	}
	rbm.setParameterVector(params);
	
	MarkovChain<GibbsOperator<BinaryRBM> > chain(&rbm);
	chain.setBatchSize(batchSize);
	chain.initializeChain(RealMatrix(batchSize,4,0));
	chain.step(1000);//burn in
	
	//evaluate distribution for all beta values
	RealVector pHidden =  exp(rbm.energy().logUnnormalizedProbabilityHidden(stateMatrix,blas::repeat(1.0,16)));
	RealVector pVisible =  exp(rbm.energy().logUnnormalizedProbabilityVisible(stateMatrix,blas::repeat(1.0,16)));
	//normalize to 1
	pHidden /= sum(pHidden);
	pVisible /= sum(pVisible);
	
	RealVector pHiddenHist(16,0.0);
	RealVector pVisibleHist(16,0.0);
	
	for(std::size_t s = 0; s != numSamples; ++s){
		chain.step(1);
		//get state number for every sampled state and add the sample to the histogram
		for(std::size_t k = 0; k != batchSize; ++k){
			std::size_t stateH = 0;
			std::size_t stateV = 0;
			for(std::size_t i = 0; i != 4; ++i){
				stateH += chain.samples().hidden.state(k,i) > 0? (1<<i):0; 
				stateV += chain.samples().visible.state(k,i) > 0? (1<<i):0; 
			}
			pHiddenHist(stateH)+=1.0/(numSamples*batchSize);
			pVisibleHist(stateV)+=1.0/(numSamples*batchSize);
		}
	}
	//calculate KL divergence between distributions
	double KLV= sum(pVisible*log(pVisible/pVisibleHist));
	double KLH= sum(pHidden*log(pHidden/pHiddenHist));
	std::cout<<KLV <<" "<< KLH<<"\n";
	BOOST_CHECK_SMALL(KLV,0.01);
	BOOST_CHECK_SMALL(KLH,0.01);
}
