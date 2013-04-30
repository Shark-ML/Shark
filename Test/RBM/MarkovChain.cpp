#include <shark/Unsupervised/RBM/Sampling/MarkovChain.h>

#define BOOST_TEST_MODULE RBM_MarkovChain
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


//this is a mockup test, meaning we implement a failsafe operator to test the markov chain without noise!

//a correct step of this will lead to: hidden=visible*2+1 and visible = hidden*3+5
class Operator{
public:
	typedef int HiddenSample;
	typedef int VisibleSample;
	struct RBM{
		std::size_t numberOfHN()const{
			return 0;
		}
		std::size_t numberOfVN()const{
			return 0;
		}
		typedef int VectorType;
	};

	RBM* mpe_rbm;

	Operator(RBM* rbm):mpe_rbm(rbm){}

	void precomputeHidden(IntVector& hidden,const IntVector& visible,double beta = 1.0)const{
		hidden(0) = 2 * visible(0);

	}
	///\brief calculates the statistics of the visible units
	void precomputeVisible(const IntVector& hidden, IntVector& visible,double beta = 1.0)const{
		visible(0) = 3 * hidden(0);
	}
	
	///\brief samples the state of the hidden units using the precomputed statistics
	void sampleHidden(IntVector& sample,double beta = 1.0)const{
		sample(0)+=1;
	}
	///\brief samples the visible units using the precomputed statistics
	void sampleVisible(IntVector& sample,double beta = 1.0)const{
		sample(0) +=5;
	}

	///\brief creates a hidden/visible sample pair from a sample. this can directly be used to calculate the gradient
	void createSample(IntVector& hidden,IntVector& visible, const IntVector& state, double beta = 1.0)const{
		BOOST_REQUIRE_EQUAL(hidden.size(),1);
		BOOST_REQUIRE_EQUAL(visible.size(),1);
		hidden(0) = 1;
		visible(0) = 1000;
	}
	
	RealVector calculateEnergy(const IntVector& hidden, const IntVector& visible){
		RealVector result(1);
		result(0)= hidden(0)+visible(0);
		return result;
	}
	
	RBM* rbm()const{
		return mpe_rbm;
	}
};

BOOST_AUTO_TEST_CASE( MarkovChain_TestVH )
{
	Operator::RBM rbmMockup;
	MarkovChain<Operator> chain(&rbmMockup);
	//chain.setVHChain();
	
	//test the created sample for equality
	//if this fails, there is no sense in going on
	chain.setBatchSize(1);
	IntVector state(1);
	state(0)=0;
	chain.initializeChain(state);	
	BOOST_REQUIRE_EQUAL(chain.sample().hidden,1u);
	BOOST_REQUIRE_EQUAL(chain.sample().visible,1000u);
	
	//now do a single step and test the result
	chain.step(1);
	BOOST_CHECK_EQUAL(chain.sample().visible,11u);
	BOOST_CHECK_EQUAL(chain.sample().hidden,22u);
	
	BOOST_CHECK_SMALL(chain.sample().energy-33,1e-10);
	
	//and now check multiple steps
	chain.step(2);
	
	//calculate result
	int hidden = 23;
	int visible = 11;
	
	//step 1
	visible = hidden * 3 +5;
	hidden = visible * 2 +1;
	
	//step 2
	visible = hidden * 3 +5;
	hidden = visible * 2;
	
	
	//test
	BOOST_CHECK_EQUAL(chain.sample().hidden,hidden);
	BOOST_CHECK_EQUAL(chain.sample().visible,visible);
	BOOST_CHECK_SMALL(chain.sample().energy-hidden-visible,1e-10);
	
}
