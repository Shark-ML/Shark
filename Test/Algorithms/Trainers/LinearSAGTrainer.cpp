//===========================================================================
/*!
 * 
 *
 * \brief       test case for the Linear SAG-Trainer
 * 
 * 
 * 
 *
 * \author      O.Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_Linear_SAG_Trainer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/LinearSAGTrainer.h>
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/ObjectiveFunctions/Regularizer.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Linear_SAG_Trainer)


template<class Dataset>
void testClassification(Dataset const& dataset, double lambda, unsigned int epochs, bool trainOffset){
	CrossEntropy<unsigned int, RealVector> loss;
	LinearClassifier<RealVector> model;
	
	
	LinearSAGTrainer<RealVector,unsigned int> trainer(&loss);
	BOOST_CHECK_EQUAL(trainer.lambda(),0.0);
	BOOST_CHECK_EQUAL(trainer.trainOffset(),true);
	
	trainer.setLambda(lambda);
	trainer.setEpochs(epochs);
	//~ trainer.setLearningRate(0.1);
	trainer.setTrainOffset(trainOffset);
	BOOST_CHECK_CLOSE(trainer.lambda(),lambda,1.e-10);
	BOOST_CHECK_EQUAL(trainer.epochs(),epochs);
	//~ BOOST_CHECK_CLOSE(trainer.learningRate(),0.1,1.e-10);
	BOOST_CHECK_EQUAL(trainer.trainOffset(),trainOffset);
	
	trainer.train(model, dataset);
	RealVector params = model.parameterVector();
	std::size_t classes = numberOfClasses(dataset);
	if(classes == 2) classes = 1;
	BOOST_REQUIRE_EQUAL(params.size(), classes * (inputDimension(dataset) +trainOffset));
	
	ErrorFunction<>error(dataset, &model.decisionFunction(), &loss);
	TwoNormRegularizer<> regularizer;
	if(trainOffset){
		RealVector mask(params.size(),1);
		subrange(mask,mask.size()-classes,mask.size()).clear();//no punishing of offset parameters
		regularizer.setMask(mask);
	}
	error.setRegularizer(lambda,&regularizer);
	
	RealVector grad;
	error.evalDerivative(params,grad);
	BOOST_CHECK_SMALL(norm_inf(grad),1.e-5);
}
template<class Dataset>
void testRegression(Dataset const& dataset, double lambda, unsigned int epochs, bool trainOffset){
	SquaredLoss<> loss;
	LinearModel<RealVector> model;
	
	
	LinearSAGTrainer<RealVector,RealVector> trainer(&loss);
	BOOST_CHECK_EQUAL(trainer.lambda(),0.0);
	BOOST_CHECK_EQUAL(trainer.trainOffset(),true);
	
	trainer.setLambda(lambda);
	trainer.setEpochs(epochs);
	//~ trainer.setLearningRate(0.01);
	trainer.setTrainOffset(trainOffset);
	BOOST_CHECK_CLOSE(trainer.lambda(),lambda,1.e-10);
	BOOST_CHECK_EQUAL(trainer.epochs(),epochs);
	//~ BOOST_CHECK_CLOSE(trainer.learningRate(),0.01,1.e-10);
	BOOST_CHECK_EQUAL(trainer.trainOffset(),trainOffset);
	
	trainer.train(model, dataset);
	RealVector params = model.parameterVector();
	BOOST_REQUIRE_EQUAL(params.size(), labelDimension(dataset) *(inputDimension(dataset) +trainOffset));
	
	ErrorFunction<> error(dataset, &model, &loss);
	TwoNormRegularizer<> regularizer;
	if(trainOffset){
		RealVector mask(params.size(),1);
		subrange(mask,mask.size()-labelDimension(dataset),mask.size()).clear();//no punishing of offset parameters
		regularizer.setMask(mask);
	}
	error.setRegularizer(lambda,&regularizer);
	
	RealVector grad;
	error.evalDerivative(params,grad);
	BOOST_CHECK_SMALL(norm_inf(grad),1.e-5);
}
BOOST_AUTO_TEST_CASE( Linear_SAG_Trainer_Test_2Classes_Offset )
{
	// simple  dataset
	Chessboard problem;
	ClassificationDataset dataset = problem.generateDataset(30);
	std::cout<<"train 2 classes with offset"<<std::endl;
	testClassification(dataset,0.1,1000,true);
	std::cout<<"train 2 classes without offset"<<std::endl;
	testClassification(dataset,0.1,1000,false);
	
	RealVector weights(dataset.numberOfElements());
	for(auto& weight: weights)
		weight = random::uni(random::globalRng,0.2,4);
	
	WeightedLabeledData<RealVector,unsigned int> weightedDataset(dataset,createDataFromRange(weights));
	std::cout<<"train 2 classes weighted with offset"<<std::endl;
	testClassification(weightedDataset,0.1,1000,true);
	std::cout<<"train 2 classes weighted without offset"<<std::endl;
	testClassification(weightedDataset,0.1,1000,false);
	
}

BOOST_AUTO_TEST_CASE( Linear_SAG_Trainer_Test_Regression)
{
	const size_t trainExamples = 30;
	LinearModel<> model;
	RealMatrix matrix(2, 2);
	RealVector offset(2);
	matrix(0,0) = 3;
	matrix(1,1) = -5;
	matrix(0,1) = -2;
	matrix(1,0) = 7;
	offset(0) = 3;
	offset(1) = -6;
	model.setStructure(matrix, offset);

	// create datatset - the model output + Gaussian noise
	RealMatrix covariance(2, 2);
	covariance(0,0) = 1;
	covariance(0,1) = 0;
	covariance(1,0) = 0;
	covariance(1,1) = 1;
	MultiVariateNormalDistribution noise(covariance);

	// create samples
	std::vector<RealVector> input(trainExamples,RealVector(2));
	std::vector<RealVector> trainTarget(trainExamples,RealVector(2));
	for (size_t i=0;i!=trainExamples;++i) {
		input[i](0) = random::uni(random::globalRng,-3.0,3.0);
		input[i](1) = random::uni(random::globalRng,-3.0,3.0);
		trainTarget[i] = noise(random::globalRng).first + model(input[i]);
	}
	RegressionDataset dataset = createLabeledDataFromRange(input, trainTarget);
	
	testRegression(dataset,0.1,100,true);
	testRegression(dataset,0.1,100,false);
	
	RealVector weights(dataset.numberOfElements());
	for(auto& weight: weights)
		weight = random::uni(random::globalRng,0.2,4);
	
	WeightedLabeledData<RealVector,RealVector> weightedDataset(dataset,createDataFromRange(weights));
	testRegression(weightedDataset,0.1,100,true);
	testRegression(weightedDataset,0.1,100,false);
	
}

BOOST_AUTO_TEST_SUITE_END()
