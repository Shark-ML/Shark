#define BOOST_TEST_MODULE Models_OnlineRNNet
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/OnlineRNNet.h>
#include <sstream>


#include <shark/Core/Random.h>

using namespace std;
using namespace boost::archive;
using namespace shark;


class OnlineRNNetTestHelper: public AbstractModel<RealVector, RealVector>{
public:
	OnlineRNNetTestHelper(std::size_t seqLength, bool bias)
	: m_structure(OnlineRNNetTestHelper::createStructure(bias))
	, m_network(&m_structure,true)
	, m_seqLength(seqLength){
		m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	}
	
	//! get internal parameters of the model
	RealVector parameterVector() const{
		return m_network.parameterVector();
	}
	//! set internal parameters of the model
	void setParameterVector(RealVector const& newParameters){
		m_network.setParameterVector(newParameters);
	}

	//!number of parameters of the network
	std::size_t numberOfParameters() const{
		return m_network.numberOfParameters();
	}
	
	boost::shared_ptr<State> createState()const{
		return m_network.createState();
	}
	
	std::size_t inputSize()const{
		return m_seqLength * m_network.inputSize();
	}
	
	std::size_t outputSize()const{
		return m_network.outputSize();
	}
	
	void eval(RealMatrix const& patterns,RealMatrix& output, State& state)const{
		for(std::size_t i = 0; i != m_seqLength; ++i){
			RealMatrix pattern = columns(patterns,i*m_structure.inputs(),(i+1)*m_structure.inputs());
			m_network.eval(pattern,output,state);
		}
	}
	
	void weightedParameterDerivative(
		RealMatrix const& patterns,RealMatrix const& outputs, RealMatrix const& coefficients,
		State const& state, RealVector& gradient
	)const{
		std::size_t start = patterns.size2() - m_structure.inputs();
		RealMatrix pattern = columns(patterns,start,patterns.size2());
		m_network.weightedParameterDerivative(pattern, outputs, coefficients,state,gradient);
	}
	
private:
	static RecurrentStructure createStructure(bool bias){
		RecurrentStructure structure;
		structure.setStructure(2,4,2,bias);
		return structure;
	}
	RecurrentStructure m_structure;
	OnlineRNNet m_network;
	std::size_t m_seqLength;
};


BOOST_AUTO_TEST_SUITE (Models_OnlineRNNet)

//test the case when the sequence has length 1. this is
//an easy case as the effect of the old gradient step is 0
//(similar to a network without hidden layer and nonlinear outputs)
BOOST_AUTO_TEST_CASE(WeightedDerivatives_Single)
{
	{
		OnlineRNNetTestHelper net(1,true);
		BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 9*6);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
	}
	{
		OnlineRNNetTestHelper net(1,false);
		BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 8*6);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
	}
}

//This is the more complex case as the effect of the first iteration is now 
//fed into the next step
//(similar to a network with one hidden layer)
BOOST_AUTO_TEST_CASE(WeightedDerivatives_TwoStep)
{
	{
		OnlineRNNetTestHelper net(2,true);
		BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 9*6);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
	}
	{
		OnlineRNNetTestHelper net(2,false);
		BOOST_REQUIRE_EQUAL(net.numberOfParameters(), 8*6);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
	}
}

//This is the more complex case as the effect of the first iteration is now 
//fed into the next step
//(similar to a network with one hidden layer)
BOOST_AUTO_TEST_CASE(WeightedDerivatives_Multiple)
{
	OnlineRNNetTestHelper net(4,true);
	testWeightedDerivative(net,1000,5.e-6,1.e-7);
}

BOOST_AUTO_TEST_SUITE_END()
