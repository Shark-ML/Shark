#define BOOST_TEST_MODULE MODEL_AUTOENCODER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/Autoencoder.h>
#include <sstream>
#include <boost/archive/polymorphic_text_iarchive.hpp>
#include <boost/archive/polymorphic_text_oarchive.hpp>
#include <shark/Rng/GlobalRng.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_AUTO_TEST_CASE( AUTOENCODER_Structure)
{
	std::size_t weightNum = 2*2*3+5;
	Autoencoder<LogisticNeuron> net;
	net.setStructure(2,3);
	BOOST_REQUIRE_EQUAL(net.hiddenBias().size(),3u);
	BOOST_REQUIRE_EQUAL(net.outputBias().size(),2u);
	BOOST_CHECK_EQUAL(net.encoderMatrix().size1(), 3u);
	BOOST_CHECK_EQUAL(net.encoderMatrix().size2(), 2u);
	BOOST_CHECK_EQUAL(net.decoderMatrix().size1(), 2u);
	BOOST_CHECK_EQUAL(net.decoderMatrix().size2(), 3u);
	BOOST_REQUIRE_EQUAL(net.numberOfParameters(),weightNum);
	
	RealVector newParams(weightNum);
	for(std::size_t i = 0; i != weightNum; ++i){
		newParams(i) = Rng::uni(0,1);
	}
	//check that setting and getting parameters works
	net.setParameterVector(newParams);
	RealVector params = net.parameterVector();
	BOOST_REQUIRE_EQUAL(params.size(),newParams.size());
	for(std::size_t i = 0; i != weightNum; ++i){
		BOOST_CHECK_CLOSE(newParams(i),params(i), 1.e-10);
	}
	//check that the weight matrix has the right values
	std::size_t param = 0;
	for(std::size_t i = 0; i != net.encoderMatrix().size1(); ++i){
		for(std::size_t j = 0; j != net.encoderMatrix().size2(); ++j,++param){
			BOOST_CHECK_EQUAL(net.encoderMatrix()(i,j), newParams(param));
		}
	}
	for(std::size_t i = 0; i != net.decoderMatrix().size1(); ++i){
		for(std::size_t j = 0; j != net.decoderMatrix().size2(); ++j,++param){
			BOOST_CHECK_EQUAL(net.decoderMatrix()(i,j), newParams(param));
		}
	}
	for(std::size_t i = 0; i != 3; ++i,++param){
		BOOST_CHECK_EQUAL(net.hiddenBias()(i), newParams(param));
	}
	BOOST_CHECK_EQUAL(net.outputBias()(0), newParams(param));
	BOOST_CHECK_EQUAL(net.outputBias()(1), newParams(param+1));
}

BOOST_AUTO_TEST_CASE( AUTOENCODER_Value )
{
	Autoencoder<LogisticNeuron> net;
	net.setStructure(3,2);
	std::size_t numParams = 2*3*2+5;
	
	for(std::size_t i = 0; i != 100; ++i){
		//initialize parameters
		RealVector parameters(numParams);
		for(size_t j=0; j != numParams;++j)
			parameters(j)=Rng::gauss(0,1);
		net.setParameterVector(parameters);

		//the testpoints
		RealVector point(3);
		point(0)=Rng::uni(-5,5);
		point(1)= Rng::uni(-5,5);
		point(2)= Rng::uni(-5,5);

		//evaluate ground truth result
		RealVector hidden = sigmoid(prod(net.encoderMatrix(),point)+net.hiddenBias());
		RealVector output = prod(net.decoderMatrix(),hidden)+net.outputBias();
		
		//check whether final result is correct
		RealVector netResult = net(point);
		BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
		BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
		BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
	}
	
	//now also test batches
	RealMatrix inputs(100,3);
	for(std::size_t i = 0; i != 100; ++i){
		inputs(i,0)=Rng::uni(-5,5);
		inputs(i,1)= Rng::uni(-5,5);
		inputs(i,2)= Rng::uni(-5,5);
	}
	testBatchEval(net,inputs);
}

BOOST_AUTO_TEST_CASE( AUTOENCODER_WeightedDerivatives)
{
	Autoencoder<TanhNeuron> net;
	net.setStructure(2,5);

	testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
	testWeightedDerivative(net,1000,5.e-6,1.e-7);
	testWeightedDerivativesSame(net,1000);
}
