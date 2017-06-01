#define BOOST_TEST_MODULE MODEL_NEURONLAYER
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/NeuronLayers.h>

using namespace shark;

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_AUTO_TEST_SUITE (Models_DropoutLayer)


typedef boost::mpl::list<TanhNeuron,LinearNeuron, LogisticNeuron, FastSigmoidNeuron> neuron_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(DropoutLayer_Value, Neuron,neuron_types){
	NeuronLayer<Neuron> layer(10);
	RealMatrix inputs(100,10);
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
		}
	}
	testBatchEval(layer,inputs);
	
}
BOOST_AUTO_TEST_CASE_TEMPLATE(DropoutLayer_Derivative, Neuron,neuron_types) {
	NeuronLayer<Neuron> layer(10);
	RealMatrix inputs(100,10);
	for(std::size_t i = 0; i != 100; ++i){
		for(std::size_t j = 0; j != 10; ++j){
			inputs(i,j) = random::uni(random::globalRng,2,3);
		}
	}
	testWeightedInputDerivative(layer);
	
}

BOOST_AUTO_TEST_SUITE_END()
