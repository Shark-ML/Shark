#define BOOST_TEST_MODULE ML_FFNET
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "derivativeTestHelper.h"

#include <shark/Models/FFNet.h>
#include <sstream>


#include <shark/Rng/GlobalRng.h>

using namespace std;
using namespace boost::archive;
using namespace shark;

double activation(double a)
{
	return 1.0/(1.0+std::exp(-a));
}

//check that the structure is correct, i.e. matrice have the right form and setting parameters works
BOOST_AUTO_TEST_SUITE (Models_FFNet)

BOOST_AUTO_TEST_CASE( FFNET_structure_Normal)
{
	//no bias
	{
		std::size_t weightNum = 2*3+3*4;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::Normal,false);
		BOOST_REQUIRE_EQUAL(net.bias().size(),0u);// no bias!
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);
		
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(), 0u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 0u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 2; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
					BOOST_CHECK_EQUAL(net.backpropMatrices()[k](j,i), net.layerMatrices()[k](i,j));
				}
			}
		}
	}
	
	//no bias, 3 layers
	{
		std::size_t weightNum = 2*3+3*4+4*5;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::Normal,false);
		BOOST_REQUIRE_EQUAL(net.bias().size(),0u);// no bias!
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size1(), 5u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size2(), 4u);
		
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(), 0u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 0u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size1(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size2(), 5u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 3; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
					BOOST_CHECK_EQUAL(net.backpropMatrices()[k](j,i), net.layerMatrices()[k](i,j));
				}
			}
		}
	}
	
	//with bias
	{
		std::size_t weightNum = 2*3+3*4+7;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::Normal,true);
		BOOST_REQUIRE_EQUAL(net.bias().size(),7u);
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);
		
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(),0u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 0u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 2; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
					BOOST_CHECK_EQUAL(net.backpropMatrices()[k](j,i), net.layerMatrices()[k](i,j));
				}
			}
		}
		for(std::size_t i = 0; i != 7; ++i){
			BOOST_CHECK_EQUAL(net.bias()(i), newParams(weightNum-7+i));
		}
	}
	
}

BOOST_AUTO_TEST_CASE( FFNET_structure_InputOutputShortcut)
{
	//for this test, we add another layer as the 3 layer version is redirected to Full- the next test
	//no bias, 3 layers
	{
		std::size_t weightNum = 2*3+3*4+4*5+2*5;
		std::size_t shortcutStart = 2*3+3*4+4*5;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::InputOutputShortcut,false);
		BOOST_REQUIRE_EQUAL(net.bias().size(),0u);// no bias!
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size1(), 5u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size2(), 4u);
		
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(), 5u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 2u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size1(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size2(), 5u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 3; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
					BOOST_CHECK_EQUAL(net.backpropMatrices()[k](j,i), net.layerMatrices()[k](i,j));
				}
			}
		}
		
		//check shortcuts
		std::size_t pos = shortcutStart;
		for(std::size_t i = 0; i != 4; ++i){
			for(std::size_t j = 0; j != 2; ++j,++pos){
				BOOST_CHECK_EQUAL(net.inputOutputShortcut()(i,j), newParams(pos));
			}
		}
	}
	
	//no bias, two layers
	{
		std::size_t weightNum = 2*3+5*4;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::InputOutputShortcut,false);
		BOOST_REQUIRE_EQUAL(net.bias().size(),0u);
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 5u);
		
		//shortcut is implemented by using the usual layerMatrices
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(), 0u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 0u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 7u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 2; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
				}
			}
		}
		
		//check backprop matrices
		//weights from layer 1 to layer 0
		for(std::size_t i = 0; i != net.layerMatrices()[0].size1(); ++i){
			for(std::size_t j = 0; j != net.layerMatrices()[0].size2(); ++j){
				BOOST_CHECK_EQUAL(net.backpropMatrices()[0](j,i), net.layerMatrices()[0](i,j));
			}
		}
		//weights from layer 2 to layer 0
		for(std::size_t i = 0; i != net.layerMatrices()[1].size1(); ++i){
			for(std::size_t j = 0; j != net.layerMatrices()[0].size2(); ++j){
				BOOST_CHECK_EQUAL(net.backpropMatrices()[0](j,i+3), net.layerMatrices()[1](i,j));
			}
		}
		//weights from layer 2 to layer 1
		for(std::size_t i = 0; i != net.layerMatrices()[1].size1(); ++i){
			for(std::size_t j = 0; j != 3; ++j){
				BOOST_CHECK_EQUAL(net.backpropMatrices()[1](j,i), net.layerMatrices()[1](i,j+2));
			}
		}
	}
	
	//with bias, 3 layers
	{
		std::size_t weightNum = 2*3+3*4+4*5+2*5+12;
		std::size_t shortcutStart = 2*3+3*4+4*5+12;
		std::size_t biasStart = 2*3+3*4+4*5;
		FFNet<LogisticNeuron,LogisticNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::InputOutputShortcut,true);
		BOOST_REQUIRE_EQUAL(net.bias().size(),12u);// no bias!
		BOOST_REQUIRE_EQUAL(net.layerMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size1(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[0].size2(), 2u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size1(), 4u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[1].size2(), 3u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size1(), 5u);
		BOOST_CHECK_EQUAL(net.layerMatrices()[2].size2(), 4u);
		
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size1(), 5u);
		BOOST_CHECK_EQUAL(net.inputOutputShortcut().size2(), 2u);
		
		BOOST_REQUIRE_EQUAL(net.backpropMatrices().size(),3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size1(), 2u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[0].size2(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size1(), 3u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[1].size2(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size1(), 4u);
		BOOST_CHECK_EQUAL(net.backpropMatrices()[2].size2(), 5u);
		
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
		//check that the weight matrices have the right values
		std::size_t param = 0;
		for(std::size_t k = 0; k != 3; ++k){
			for(std::size_t i = 0; i != net.layerMatrices()[k].size1(); ++i){
				for(std::size_t j = 0; j != net.layerMatrices()[k].size2(); ++j,++param){
					BOOST_CHECK_EQUAL(net.layerMatrices()[k](i,j), newParams(param));
					BOOST_CHECK_EQUAL(net.backpropMatrices()[k](j,i), net.layerMatrices()[k](i,j));
				}
			}
		}
		
		//check shortcuts
		std::size_t pos = shortcutStart;
		for(std::size_t i = 0; i != 5; ++i){
			for(std::size_t j = 0; j != 2; ++j,++pos){
				BOOST_CHECK_EQUAL(net.inputOutputShortcut()(i,j), newParams(pos));
			}
		}
		
		for(std::size_t i = 0; i != 7; ++i){
			BOOST_CHECK_EQUAL(net.bias()(i), newParams(biasStart+i));
		}
	}
}
//todo: test Full (the simple case with 2 layers and shortcut input/output is already tested)


// given that the structure is correct, we can now test eval by giving it random parameters
// and random inputs and compute the result by hand. This test is quite long as we have to
// check a lot of different structures. Also for every structure we ensure that all calls to eval
// produce the same output given a set of inputs
BOOST_AUTO_TEST_CASE( FFNET_Value )
{
	//2 layers, no shortcut, no bias
	{
		FFNet<LogisticNeuron,LinearNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::Normal,false);
		std::size_t numParams = 2*3+3*4;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden = sigmoid(prod(net.layerMatrices()[0],point));
			RealVector output = prod(net.layerMatrices()[1],hidden);
			
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
	//2 layers, no shortcut, bias
	{
		FFNet<LogisticNeuron,LinearNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::Normal,true);
		std::size_t numParams = 2*3+3*4+7;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden = sigmoid(prod(net.layerMatrices()[0],point)+subrange(net.bias(),0,3));
			RealVector output = prod(net.layerMatrices()[1],hidden)+subrange(net.bias(),3,7);
			
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
	//2 layers, shortcut, bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,3,4,FFNetStructures::InputOutputShortcut,true);
		std::size_t numParams = 2*3+3*4+2*4+7;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden = sigmoid(prod(net.layerMatrices()[0],point)+subrange(net.bias(),0,3));
			RealVector output = prod(columns(net.layerMatrices()[1],2,5),hidden);
			output += prod(columns(net.layerMatrices()[1],0,2),point);
			output += subrange(net.bias(),3,7);
			output = tanh(output);
			
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
	//3 layers, no shortcut
	{
		FFNet<LogisticNeuron,LinearNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::Normal,false);
		std::size_t numParams = 2*3+3*4+4*5;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden1 = sigmoid(prod(net.layerMatrices()[0],point));
			RealVector hidden2 = sigmoid(prod(net.layerMatrices()[1],hidden1));
			RealVector output = prod(net.layerMatrices()[2],hidden2);
			
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
			BOOST_CHECK_SMALL(netResult(4)-output(4),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
	//3 layers, shortcut
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::InputOutputShortcut,false);
		std::size_t numParams = 2*3+3*4+4*5+2*5;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden1 = sigmoid(prod(net.layerMatrices()[0],point));
			RealVector hidden2 = sigmoid(prod(net.layerMatrices()[1],hidden1));
			RealVector output = prod(net.layerMatrices()[2],hidden2)
				+ prod(net.inputOutputShortcut(),point);
			output =tanh(output);
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
			BOOST_CHECK_SMALL(netResult(4)-output(4),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
	//3 layers, full
	{
		FFNet<LogisticNeuron,LinearNeuron> net;
		net.setStructure(2,3,4,5,FFNetStructures::Full,false);
		std::size_t numParams = 2*3+3*4+4*5+2*5+2*4+3*5;
		
		for(std::size_t i = 0; i != 100; ++i){
			//initialize parameters
			RealVector parameters(numParams);
			for(size_t j=0; j != numParams;++j)
				parameters(j)=Rng::gauss(0,1);
			net.setParameterVector(parameters);

			//the testpoints
			RealVector point(2);
			point(0)=Rng::uni(-5,5);
			point(1)= Rng::uni(-5,5);

			//evaluate ground truth result
			RealVector hidden1 = sigmoid(prod(net.layerMatrices()[0],point));
			
			RealVector hidden2 = prod(columns(net.layerMatrices()[1],0,2),point);
			hidden2 += prod(columns(net.layerMatrices()[1],2,5),hidden1);
			hidden2 = sigmoid(hidden2);
			
			RealVector output = prod(columns(net.layerMatrices()[2],0,2),point);
			output += prod(columns(net.layerMatrices()[2],2,5),hidden1);
			output += prod(columns(net.layerMatrices()[2],5,9),hidden2);
			
			//check whether final result is correct
			RealVector netResult = net(point);
			BOOST_CHECK_SMALL(netResult(0)-output(0),1.e-12);
			BOOST_CHECK_SMALL(netResult(1)-output(1),1.e-12);
			BOOST_CHECK_SMALL(netResult(2)-output(2),1.e-12);
			BOOST_CHECK_SMALL(netResult(3)-output(3),1.e-12);
			BOOST_CHECK_SMALL(netResult(4)-output(4),1.e-12);
		}
		
		//now also test batches
		RealMatrix inputs(100,2);
		for(std::size_t i = 0; i != 100; ++i){
			inputs(i,0)=Rng::uni(-5,5);
			inputs(i,1)= Rng::uni(-5,5);
		}
		testBatchEval(net,inputs);
	}
	
}

BOOST_AUTO_TEST_CASE( FFNET_WeightedDerivatives)
{
	//2 layers, Normal, no Bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,FFNetStructures::Normal,false);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	//2 layers, Normal, bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,FFNetStructures::Normal,true);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	//2 layers, Shortcut, bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,FFNetStructures::InputOutputShortcut,true);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	
	//2 layers, Full, bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,FFNetStructures::Full,true);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	//3 layers, Normal, no Bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,3,FFNetStructures::Normal,false);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	
	//3 layers, Shortcut, Bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,3,FFNetStructures::InputOutputShortcut,true);

		testWeightedInputDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivative(net,1000,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,1000);
	}
	
	//3 layers, Full, Bias
	{
		FFNet<LogisticNeuron,TanhNeuron> net;
		net.setStructure(2,5,3,3,FFNetStructures::Full,true);

		testWeightedInputDerivative(net,100,5.e-6,1.e-7);
		testWeightedDerivative(net,100,5.e-6,1.e-7);
		testWeightedDerivativesSame(net,100);
	}
}

BOOST_AUTO_TEST_SUITE_END()
