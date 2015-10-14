#define BOOST_TEST_MODULE ML_SIGMOID_FIT
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/SigmoidFit.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Models/Converter.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_SigmoidFit)

BOOST_AUTO_TEST_CASE( SIGMOID_FIT_TEST_RPROP_NO_ENCODING_DETERMINISTIC ){
	bool TRANSFORM_INPUTS = false; //no encoding, i.e., the non-unconstrained variant
	unsigned int NUM_DRAWS = 90;
	// create model, and vars for the flow
	RealVector gen_parameters( 2 ); //for generating data
	RealVector obscure_parameters( 2 ); //starting points before optimization/training

	// generate random parameter set
	gen_parameters(0) = 1.4;
	gen_parameters(1) = -0.3;
	
	// for probabilities of 0.1, ... , 0.9 , sample from the model
	std::vector<RealVector> dataSamplePoints(NUM_DRAWS,RealVector(1));
	for ( size_t j=0; j<9; j++ ) {
		double target = 0.1*(j+1); //these are the probabilities i want.
		double source = 1.0/gen_parameters(0) * (gen_parameters(1) - log(-1+1/target)); //and i have to look here on the x-axis to get them.
		for ( unsigned int k=0; k<10; k++ ) {
			dataSamplePoints[j*10+k](0)=source;
		}
	}
	// set the targets according to their probabilities
	std::vector<unsigned int> dataSampleLabels(NUM_DRAWS,0);
	for ( size_t i=0; i<9; i++ ) {
		for ( std::size_t k=0; k<i+1; k++ ) {
			dataSampleLabels[ i*10+k] = 1; //set to positive class according to probability
		}
	}
	ClassificationDataset dataset = createLabeledDataFromRange(dataSamplePoints, dataSampleLabels);
	
	// now start testing phase: construct new model for safety, set to arbitrary state
	SigmoidFitRpropNLL trainer(100);
	SigmoidModel test_model( TRANSFORM_INPUTS );
	obscure_parameters(0) = 0.3;
	obscure_parameters(1) = 1.0;
	test_model.setParameterVector( obscure_parameters );
	trainer.train( test_model, dataset );

	// check the parameters
	RealVector estimate = test_model.parameterVector();
	BOOST_CHECK_SMALL( estimate(0) - gen_parameters(0), 1E-9);
	BOOST_CHECK_SMALL( estimate(1) - gen_parameters(1), 1E-9);
}

BOOST_AUTO_TEST_CASE( SIGMOID_FIT_TEST_RPROP_WITH_ENCODING_DETERMINISTIC ){
	bool TRANSFORM_INPUTS = true; //unconstrained variant
	unsigned int NUM_DRAWS = 90;
	// create model, and vars for the flow
	RealVector gen_parameters( 2 ); //for generating data
	RealVector obscure_parameters( 2 ); //starting points before optimization/training

	// generate random parameter set
	gen_parameters(0) = 1.4;
	gen_parameters(1) = -0.3;
	// for probabilities of 0.1, ... , 0.9 , sample from the model
	std::vector<RealVector> dataSamplePoints(NUM_DRAWS,RealVector(1));
	for ( size_t j=0; j<9; j++ ) {
		double target = 0.1*(j+1); //these are the probabilities i want.
		double source = 1.0/exp(gen_parameters(0)) * (gen_parameters(1) - log(-1+1/target)); //and i have to look here on the x-axis to get them.
		for ( unsigned int k=0; k<10; k++ ) {
			dataSamplePoints[j*10+k](0)=source;
		}
	}
	// set the targets according to their probabilities
	std::vector<unsigned int> dataSampleLabels(NUM_DRAWS,0);
	for ( size_t i=0; i<9; i++ ) {
		for ( std::size_t k=0; k<i+1; k++ ) {
			dataSampleLabels[ i*10+k] = 1; //set to positive class according to probability
		}
	}
	ClassificationDataset dataset = createLabeledDataFromRange(dataSamplePoints, dataSampleLabels);
	
	
	// now start testing phase: construct new model for safety, set to arbitrary state
	SigmoidFitRpropNLL trainer(100);
	SigmoidModel test_model( TRANSFORM_INPUTS );
	obscure_parameters(0) = 0.3;
	obscure_parameters(1) = 1.0;
	test_model.setParameterVector( obscure_parameters );
	trainer.train( test_model, dataset );

	// check the parameters
	RealVector estimate = test_model.parameterVector();
	BOOST_CHECK_SMALL( estimate(0) - gen_parameters(0), 1E-9);
	BOOST_CHECK_SMALL( estimate(1) - gen_parameters(1), 1E-9);
}

BOOST_AUTO_TEST_CASE( SIGMOID_FIT_TEST_RPROP_NO_ENCODING_DETERMINISTIC_NOBIAS ){
	bool TRANSFORM_INPUTS = false; //no encoding, i.e., the non-unconstrained variant
	unsigned int NUM_DRAWS = 90;
	// create model, and vars for the flow
	RealVector gen_parameters( 2 ); //for generating data
	RealVector obscure_parameters( 2 ); //starting points before optimization/training

	// generate random parameter set
	gen_parameters(0) = 1.4;
	gen_parameters(1) = -0.3;
	// for probabilities of 0.1, ... , 0.9 , sample from the model
	std::vector<RealVector> dataSamplePoints(NUM_DRAWS,RealVector(1));
	for ( size_t j=0; j<9; j++ ) {
		double target = 0.1*(j+1); //these are the probabilities i want.
		double source = 1.0/exp(gen_parameters(0)) * (gen_parameters(1) - log(-1+1/target)); //and i have to look here on the x-axis to get them.
		for ( unsigned int k=0; k<10; k++ ) {
			dataSamplePoints[j*10+k](0)=source;
		}
	}
	// set the targets according to their probabilities
	std::vector<unsigned int> dataSampleLabels(NUM_DRAWS,0);
	for ( size_t i=0; i<9; i++ ) {
		for ( std::size_t k=0; k<i+1; k++ ) {
			dataSampleLabels[ i*10+k] = 1; //set to positive class according to probability
		}
	}
	ClassificationDataset dataset = createLabeledDataFromRange(dataSamplePoints, dataSampleLabels);
	// now start testing phase: construct new model for safety, set to arbitrary state
	SigmoidFitRpropNLL trainer(100);
	SigmoidModel test_model( TRANSFORM_INPUTS );

	obscure_parameters(0) = 0.3;
	obscure_parameters(1) = 1.0;
	test_model.setParameterVector( obscure_parameters );
	test_model.setOffsetActivity( false ); //clamp offset to zero
	trainer.train( test_model, dataset );

	// check the parameters
	RealVector estimate = test_model.parameterVector();
	BOOST_CHECK_SMALL( estimate(1), 1E-9);
}

BOOST_AUTO_TEST_CASE( SIGMOID_FIT_TEST_RPROP_WITH_ENCODING_DETERMINISTIC_NOBIAS ){
	bool TRANSFORM_INPUTS = true; //unconstrained variant
	unsigned int NUM_DRAWS = 90;
	// create model, and vars for the flow
	RealVector gen_parameters( 2 ); //for generating data
	RealVector obscure_parameters( 2 ); //starting points before optimization/training
	
	// generate random parameter set
	gen_parameters(0) = std::log(1.4);
	gen_parameters(1) = -0.3;
	// for probabilities of 0.1, ... , 0.9 , sample from the model
	std::vector<RealVector> dataSamplePoints(NUM_DRAWS,RealVector(1));
	for ( size_t j=0; j<9; j++ ) {
		double target = 0.1*(j+1); //these are the probabilities i want.
		double source = 1.0/exp(gen_parameters(0)) * (gen_parameters(1) - log(-1+1/target)); //and i have to look here on the x-axis to get them.
		for ( unsigned int k=0; k<10; k++ ) {
			dataSamplePoints[j*10+k](0)=source;
		}
	}
	// set the targets according to their probabilities
	std::vector<unsigned int> dataSampleLabels(NUM_DRAWS,0);
	for ( size_t i=0; i<9; i++ ) {
		for ( std::size_t k=0; k<i+1; k++ ) {
			dataSampleLabels[ i*10+k] = 1; //set to positive class according to probability
		}
	}
	ClassificationDataset dataset = createLabeledDataFromRange(dataSamplePoints, dataSampleLabels);
	// now start testing phase: construct new model for safety, set to arbitrary state

	SigmoidFitRpropNLL trainer(100);
	SigmoidModel test_model( TRANSFORM_INPUTS );
	obscure_parameters(0) = 0.3;
	obscure_parameters(1) = 1.0;
	test_model.setParameterVector( obscure_parameters );
	test_model.setOffsetActivity( false ); //clamp offset to zero
	trainer.train( test_model, dataset );
}

// NOTE THAT THIS TEST OF PLATT'S METHOD IS NOT AS EXTENSIVE AS THE ABOVE AND MORE A ROUGH ESTIMATE OF USABILITY
BOOST_AUTO_TEST_CASE( SIGMOID_FIT_TEST_PLATT ){
	// create a noisy toy dataset
	const size_t examples = 10000;
	std::vector<RealVector> input(examples);
	std::vector<unsigned int> target(examples);
	size_t i;
	for (i=0; i<examples; i++)
	{
		// classes 0 and 1 are balanced
		// values of class 0 are normally distributed with mean 0 and unit variance
		// values of class 1 are normally distributed with mean -2 and unit variance
		target[i] = Rng::coinToss() ? 1 : 0;
		input[i].resize(1);
		input[i](0) = Rng::gauss() - 2.0 * target[i];
	}
	ClassificationDataset dataset = createLabeledDataFromRange(input, target);

	// model and trainer
	SigmoidModel sigmoid;
	SigmoidFitPlatt trainer;

	// train the model on the data
	trainer.train(sigmoid, dataset);

	// check the parameters
	RealVector param = sigmoid.parameterVector();
	double  slope = param(0);
	double offset = param(1) / param(0);

	BOOST_CHECK_SMALL( slope - 1.07, 0.1);
	BOOST_CHECK_SMALL(offset - 0.95, 0.1);
}

BOOST_AUTO_TEST_SUITE_END()
