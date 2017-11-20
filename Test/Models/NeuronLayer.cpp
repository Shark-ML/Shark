#define BOOST_TEST_MODULE MODEL_NEURONLAYER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/NeuronLayers.h>

using namespace shark;

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_AUTO_TEST_SUITE (Models_DropoutLayer)


typedef boost::mpl::list<TanhNeuron,LinearNeuron, LogisticNeuron, FastSigmoidNeuron, NormalizerNeuron<>, SoftmaxNeuron<> > neuron_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(NeuronLayer_Value, Neuron,neuron_types){
	NeuronLayer<Neuron> layer(10);
	RealMatrix inputs(100,10);
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
		}
	}
	testBatchEval(layer,inputs);
	
}
BOOST_AUTO_TEST_CASE_TEMPLATE(NeuronLayer_Derivative, Neuron,neuron_types) {
	NeuronLayer<Neuron> net(10);
	RealVector coefficients(10);
	RealVector point(10);
	for(unsigned int test = 0; test != 1000; ++test){
		for(size_t i = 0; i != 10;++i){
			coefficients(i) = random::uni(random::globalRng,-5,5);
		}
		for(size_t i = 0; i != 10;++i){
			point(i) = random::uni(random::globalRng,0.1,3);
		}

		testWeightedDerivative(net, point, coefficients, 1.e-5,1.e-5);
	}
}

BOOST_AUTO_TEST_SUITE_END()
