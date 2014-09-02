#define BOOST_TEST_MODULE ObjFunct_SparseAutoencoderError
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/SparseAutoencoderError.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include "TestObjectiveFunction.h"

using namespace shark;

//allways 0
class NullLoss : public AbstractLoss<RealVector, RealVector>
{
public:
	std::string name() const
	{ return "NullLoss"; }

	double eval(BatchLabelType const&, BatchOutputType const&) const{
		return 0;
	}
	double evalDerivative(
		RealMatrix const&, 
		RealMatrix const& prediction, 
		RealMatrix& gradient
	) const {
		gradient.resize(prediction.size1(),prediction.size2());
		gradient.clear();
		return 0;
	}
};

BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Value ){
	std::size_t Inputs = 100;
	std::size_t Iterations = 100;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);;
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,25);

	Autoencoder<LogisticNeuron,TanhNeuron> model;
	model.setStructure(Dimensions,5);
	NullLoss loss;

	for(std::size_t iter = 0; iter != Iterations; ++iter){
		double roh = Rng::uni(0.1,0.9);
		double beta = Rng::uni(0.1,0.9);
		
		SparseAutoencoderError error(dataset, &model,&loss,roh,beta);
		
		//evaluate error and check that its consistent
		SparseAutoencoderError::FirstOrderDerivative derivative;
		initRandomNormal(model,1);
		double errorValue = error.eval(model.parameterVector());
		double errorDerivativeValue = error.evalDerivative(model.parameterVector(),derivative);
		BOOST_CHECK_CLOSE(errorValue,errorDerivativeValue, 0.01);
		
		//now check that the error value reported is actually correct.
		
		//evaluate the inputs for the model
		RealVector activations(5,0.0);
		boost::shared_ptr<State> state = model.createState();
		RealMatrix result;
		for(std::size_t i = 0; i != 4; ++i){
			model.eval(dataset.batch(i).input,result,*state);
			//sum hidden activations
			activations+=sum_rows(model.hiddenResponses(*state));
		}
		activations /= Inputs;
		
		//calculate KL-divergence
		double errorTest = 0;
		for(std::size_t i = 0; i != 5; ++i){
			double a = activations(i);
			errorTest += roh*std::log(roh/a);
			errorTest += (1-roh)*std::log((1-roh)/(1-a));
		}
		errorTest*=beta;
		
		BOOST_CHECK_SMALL(errorTest-errorValue,1.e-14);
	}
}

BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Value_Tied ){
	std::size_t Inputs = 100;
	std::size_t Iterations = 100;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,25);

	TiedAutoencoder<LogisticNeuron,TanhNeuron> model;
	model.setStructure(Dimensions,5);
	NullLoss loss;

	for(std::size_t iter = 0; iter != Iterations; ++iter){
		double roh = Rng::uni(0.1,0.9);
		double beta = Rng::uni(0.1,0.9);
		
		SparseAutoencoderError error(dataset, &model,&loss,roh,beta);
		
		//evaluate error and check that its consistent
		SparseAutoencoderError::FirstOrderDerivative derivative;
		initRandomNormal(model,1);
		double errorValue = error.eval(model.parameterVector());
		double errorDerivativeValue = error.evalDerivative(model.parameterVector(),derivative);
		BOOST_CHECK_CLOSE(errorValue,errorDerivativeValue, 0.01);
		
		//now check that the error value reported is actually correct.
		
		//evaluate the inputs for the model
		RealVector activations(5,0.0);
		boost::shared_ptr<State> state = model.createState();
		RealMatrix result;
		for(std::size_t i = 0; i != 4; ++i){
			model.eval(dataset.batch(i).input,result,*state);
			//sum hidden activations
			activations+=sum_rows(model.hiddenResponses(*state));
		}
		activations /= Inputs;
		
		//calculate KL-divergence
		double errorTest = 0;
		for(std::size_t i = 0; i != 5; ++i){
			double a = activations(i);
			errorTest += roh*std::log(roh/a);
			errorTest += (1-roh)*std::log((1-roh)/(1-a));
		}
		errorTest*=beta;
		
		BOOST_CHECK_SMALL(errorTest-errorValue,1.e-14);
	}
}

//tests, whether the error function is processed correctly
BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Loss ){
	std::size_t Inputs = 10;
	std::size_t Iterations = 10;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,5);

	Autoencoder<LogisticNeuron,TanhNeuron> model;
	model.setStructure(Dimensions,5);
	SquaredLoss<RealVector> loss;
	SparseAutoencoderError error(dataset,&model,&loss,0.5,0.0);
	
	double errortest = error.eval(model.parameterVector());
	BOOST_CHECK_SMALL(errortest-loss(dataset.inputs(),model(dataset.inputs())),1.e-15);
	

	for(std::size_t i = 0; i != Iterations; ++i){
		initRandomNormal(model,0.1);
		RealVector point = model.parameterVector();
		SparseAutoencoderError::FirstOrderDerivative derivative;
		RealVector estimatedDerivative = estimateDerivative(error,point,1.e-10);
		error.evalDerivative(point,derivative);
		
		double errorDer= norm_inf(estimatedDerivative - derivative);
		BOOST_CHECK_SMALL(errorDer, 1.e-4);
	}

}

BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Derivative_Loss){
	std::size_t Inputs = 100;
	std::size_t Iterations = 1000;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,25);

	Autoencoder<LogisticNeuron,LinearNeuron> model;
	model.setStructure(Dimensions,2);
	SquaredLoss<> loss;

	for(std::size_t i = 0; i != Iterations; ++i){
		double roh = Rng::uni(0.1,0.9);
		double beta = Rng::uni(0.1,0.9);
	
		SparseAutoencoderError error(dataset,&model,&loss,roh,beta);
		initRandomNormal(model,0.1);
		RealVector point = model.parameterVector();
		SparseAutoencoderError::FirstOrderDerivative derivative;
		RealVector estimatedDerivative = estimateDerivative(error,point,1.e-10);
		error.evalDerivative(point,derivative);
		double errorm= norm_inf(estimatedDerivative - derivative);
		BOOST_CHECK_SMALL(errorm, 1.e-4);
	}

}


BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Derivative){
	std::size_t Inputs = 100;
	std::size_t Iterations = 1000;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,25);

	Autoencoder<LogisticNeuron,LinearNeuron> model;
	model.setStructure(Dimensions,2);
	NullLoss loss;

	for(std::size_t i = 0; i != Iterations; ++i){
		double roh = Rng::uni(0.1,0.9);
		double beta = Rng::uni(0.1,0.9);
	
		SparseAutoencoderError error(dataset,&model,&loss,roh,beta);
		initRandomNormal(model,0.1);
		RealVector point = model.parameterVector();
		SparseAutoencoderError::FirstOrderDerivative derivative;
		RealVector estimatedDerivative = estimateDerivative(error,point,1.e-10);
		error.evalDerivative(point,derivative);
		double errorm= norm_inf(estimatedDerivative - derivative);
		BOOST_CHECK_SMALL(errorm, 1.e-4);
	}

}

BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Derivative_Tied){
	std::size_t Inputs = 100;
	std::size_t Iterations = 1000;
	std::size_t Dimensions = 5;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input,25);

	TiedAutoencoder<LogisticNeuron,TanhNeuron> model;
	model.setStructure(Dimensions,2);
	NullLoss loss;

	for(std::size_t i = 0; i != Iterations; ++i){
		double roh = Rng::uni(0.1,0.9);
		double beta = Rng::uni(0.1,0.9);
	
		SparseAutoencoderError error(dataset,&model,&loss,roh,beta);
		initRandomNormal(model,0.1);
		RealVector point = model.parameterVector();
		SparseAutoencoderError::FirstOrderDerivative derivative;
		RealVector estimatedDerivative = estimateDerivative(error,point,1.e-10);
		error.evalDerivative(point,derivative);
		
		double errorm= norm_inf(estimatedDerivative - derivative);
		BOOST_CHECK_SMALL(errorm, 1.e-4);
	}

}

//this test only tests, whether the error achieves the right mean activation
BOOST_AUTO_TEST_CASE( SparseAutoencoderError_Derivative_GradDesc)
{
	std::size_t Inputs = 100;
	std::size_t Iterations = 100;
	std::size_t Dimensions = 5;
	double roh = 0.3;
	double beta = 1.0;
	std::vector<RealVector> input(Inputs,RealVector(Dimensions));
	for(std::size_t i = 0; i != Inputs; ++i){
		for(std::size_t j = 0; j != Dimensions; ++j){
			input[i](j) = Rng::uni(-1,1);;
		}
	}
	RegressionDataset dataset = createLabeledDataFromRange(input,input);

	Autoencoder<LogisticNeuron,TanhNeuron> model;
	model.setStructure(Dimensions,5);
	initRandomNormal(model,1);
	NullLoss loss;
	SparseAutoencoderError error(dataset,&model,&loss,roh,beta);

	IRpropPlus optimizer;
	optimizer.init(error);
	
	
	//now train the network to optimize the hidden units 
	//to have target roh
	for(std::size_t i = 0; i != Iterations; ++i){
		optimizer.step(error);
		//std::cout<<optimizer.solution().value<<std::endl;
	}
	model.setParameterVector(optimizer.solution().point);
	//evaluate the inputs for the model
	RealMatrix output;
	boost::shared_ptr<State> state = model.createState();
	model.eval(dataset.batch(0).input,output,*state);
	//sum hidden activations
	RealVector activations=sum_rows(model.hiddenResponses(*state));
	activations /= Inputs;
	//check that the mean activation is correct
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK_SMALL(activations(i)-roh,1.e-5);
	}
}
