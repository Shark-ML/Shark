#include <shark/Unsupervised/RBM/Sampling/TemperedMarkovChain.h>

#define BOOST_TEST_MODULE RBM_TemepredMarkovChain
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


//this is a mockup test, meaning we implement a failsafe operator to test the tempered markov chain without noise!

//doesn't change the hidden/visible at all so that the swap can be tested.

struct RBMMockup{
public:
	typedef Rng::rng_type RngType;
	typedef RealVector VectorType;

	RngType& rng(){
		return Rng::globalRng;
	}
	
	std::size_t numberOfHN()const{
		return 0;
	}
	std::size_t numberOfVN()const{
		return 0;
	}
};
class IdentityOperator{
public:
	typedef IntVector HiddenSample;
	typedef IntVector VisibleSample;
	typedef RBMMockup RBM;
	
	RBM* m_rbm;
	SamplingFlags m_flags;

	IdentityOperator(RBM* rbm):m_rbm(rbm){}
	
	RBM* rbm(){
		return m_rbm;
	
	}
	
	SamplingFlags& flags(){
		return m_flags;
	}
	const SamplingFlags& flags()const{
		return m_flags;
	}

	void precomputeHidden(IntMatrix& hidden,const IntMatrix& visible, RealVector beta)const{
	}
	///\brief calculates the statistics of the visible units
	void precomputeVisible(const IntMatrix& hidden, IntMatrix& visible, RealVector beta)const{
	}
	
	///\brief samples the state of the hidden units using the precomputed statistics
	void sampleHidden(IntMatrix& sample,double beta = 1.0)const{
	}
	///\brief samples the visible units using the precomputed statistics
	void sampleVisible(IntMatrix& sample,double beta = 1.0)const{
	}

	///\brief creates a hidden/visible sample pair from a sample. this can directly be used to calculate the gradient
	void createSample(IntMatrix& hidden,IntMatrix& visible, const RealMatrix& state, RealVector beta)const{
		visible = state;
	}
	
	///\brief samples the state of the hidden units using the precomputed statistics
	void updateHidden(IntMatrix& sample,double beta = 1.0)const{
	}
	///\brief samples the visible units using the precomputed statistics
	void updateVisible(IntMatrix& sample,double beta = 1.0)const{
	}
	
	//energy is the state of visible
	RealVector calculateEnergy(const IntMatrix& hidden, const IntMatrix& visible){
		return RealVector(column(visible,0));
	}
	
	
};

BOOST_AUTO_TEST_CASE( TemperedMarkovChain_TestSwapEnergy )
{
	RBMMockup rbmMockup;
	TemperedMarkovChain<IdentityOperator> chain(&rbmMockup);
	chain.setNumberOfTemperatures(5);
	RealMatrix state(5,1);
	for(std::size_t i = 0; i != 5; ++i){
		state(i,0) = 400-i*100;
		chain.setBeta(i,1-0.2*i);
	}
	chain.initializeChain(state);
	//now sample a lot of times and check whether the order is correct.
	//this test will fail with a remarcably low chance like 1.e-200 or so
	for(std::size_t i = 0; i != 10000;++i){
		chain.step(1);
	}
	//in the end, energies should be orderered such that chain[i].energy<chain[i+1].energy
	for(std::size_t i = 0; i != 4; ++i){
		BOOST_CHECK(chain.samples().energy(i) < chain.samples().energy(i+1));
	}
	
}
