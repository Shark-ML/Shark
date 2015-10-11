#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Rng/GlobalRng.h>
#include "TestLoss.h"

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_SQUAREDLOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;
using namespace std;

BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_SquaredLoss)

BOOST_AUTO_TEST_CASE( SQUAREDLOSS_EVAL ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		SquaredLoss<> loss;

		//sample point between -10,10
		RealMatrix testPoint(1,2);
		testPoint(0,0) = Rng::uni(-10.0,10.0);
		testPoint(0,1) = Rng::uni(-10.0,10.0);

		//sample label between -10,10
		RealMatrix testLabel(1,2);
		testLabel(0,0) = Rng::uni(-10.0,10.0);
		testLabel(0,1) = Rng::uni(-10.0,10.0);


		//the test results
		double valueResult = sqr(testPoint(0,0)-testLabel(0,0))+sqr(testPoint(0,1)-testLabel(0,1));
		RealVector estimatedDerivative = estimateDerivative(loss, testPoint, testLabel);

		//test eval
		double value = loss.eval(testLabel,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(testLabel, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( SQUAREDLOSS_EVAL_Classification ) {
	unsigned int maxTests = 10000;
	for (unsigned int test = 0; test != maxTests; ++test) {
		SquaredLoss<RealVector,unsigned int> loss;
		SquaredLoss<RealVector,RealVector> lossOneHot;

		//sample point between -10,10
		RealMatrix testPoint(1,3);
		testPoint(0,0) = Rng::uni(-10.0,10.0);
		testPoint(0,1) = Rng::uni(-10.0,10.0);
		testPoint(0,2) = Rng::uni(-10.0,10.0);

		//sample class label
		UIntVector testLabelDisc(1);
		testLabelDisc(0) = Rng::discrete(0,2);
		
		RealMatrix testLabel(1,3);
		testLabel(0,testLabelDisc(0))=1;


		//the test results
		double valueResult = lossOneHot.eval(testLabel,testPoint);
		RealVector estimatedDerivative = estimateDerivative(lossOneHot, testPoint, testLabel);

		//test eval
		double value = loss.eval(testLabelDisc,testPoint);
		BOOST_CHECK_SMALL(value-valueResult, 1.e-13);

		//test evalDerivative (first)
		RealMatrix derivative;
		value = loss.evalDerivative(testLabelDisc, testPoint, derivative);
		BOOST_CHECK_SMALL(value - valueResult, 1.e-13);
		BOOST_CHECK_SMALL(norm_2(row(derivative,0) - estimatedDerivative), 1.e-5);
	}
}

BOOST_AUTO_TEST_CASE( SQUAREDLOSS_EVAL_Sequence ) {
	unsigned int maxTests = 100;
	unsigned int dims = 10;
	unsigned int batchSize = 5;
	unsigned int sequenceLength = 30;
	for (unsigned int test = 0; test != maxTests; ++test) {
		SquaredLoss<Sequence,Sequence> loss(10);
		SquaredLoss<RealVector,RealVector> lossSingleSequence;
		
		//create the sequences as well as ground truth
		std::vector<Sequence> sequenceBatch;
		std::vector<Sequence> sequenceBatchLabel;
		std::vector<RealMatrix> sequenceBatchMat;
		std::vector<RealMatrix> sequenceBatchMatLabel;
		for(std::size_t b = 0; b != 5; ++b){
			Sequence seq(10, RealVector(dims,0.0));
			Sequence seqLabel(10, RealVector(dims,0.0));
			RealMatrix seqMat(sequenceLength-10, dims);
			RealMatrix seqMatLabel(sequenceLength-10, dims);
			
			for(std::size_t i = 0; i != sequenceLength -10; ++i){
				for(std::size_t j = 0; j != dims; ++j){
					seqMat(i,j) = Rng::gauss(0,1);
					seqMatLabel(i,j) = Rng::gauss(0,1);
				}
				seq.push_back(row(seqMat,i));
				seqLabel.push_back(row(seqMatLabel,i));
			}
			sequenceBatch.push_back(seq);
			sequenceBatchMat.push_back(seqMat);
			sequenceBatchLabel.push_back(seqLabel);
			sequenceBatchMatLabel.push_back(seqMatLabel);
		}
		
		
		//create ground truth error and gradient
		double errorTruth = 0;
		std::vector<RealMatrix> gradientTruth(batchSize);
		for(std::size_t b = 0; b != batchSize; ++b){
			errorTruth += lossSingleSequence.eval(sequenceBatchMatLabel[b],sequenceBatchMat[b]);
			lossSingleSequence.evalDerivative(sequenceBatchMatLabel[b],sequenceBatchMat[b], gradientTruth[b]);
		}
		
		double error = loss.eval(sequenceBatchLabel,sequenceBatch);
		std::vector<Sequence> gradient;
		double errorDerivative = loss.evalDerivative(sequenceBatchLabel,sequenceBatch,gradient);
		BOOST_CHECK_CLOSE(error, errorDerivative, 1.e-12);
		BOOST_CHECK_CLOSE(error, errorTruth, 1.e-10);
		
		for(std::size_t b = 0; b != batchSize; ++b){
			for(std::size_t i = 0; i != 10; ++i){
				for(std::size_t j = 0; j != dims; ++j){
					BOOST_CHECK_EQUAL(gradient[b][i][j],0.0);
				}
			}
			for(std::size_t i = 10; i != sequenceLength; ++i){
				for(std::size_t j = 0; j != dims; ++j){
					BOOST_CHECK_CLOSE(gradient[b][i][j],gradientTruth[b](i-10,j),1.e-10);
				}
			}
		}
		
	}
}

BOOST_AUTO_TEST_SUITE_END()
