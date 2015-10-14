//===========================================================================
/*!
 * 
 *
 * \brief       Test case for optimization of the hyperparameters of a
 * Gaussian Process/Regularization Network using evidence/marginal
 * likelihood maximization.
 * 
 * 
 *
 * \author      Christian Igel
 * \date        2011
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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

#include <shark/Rng/GlobalRng.h>
#include <shark/Algorithms/Trainers/RegularizationNetworkTrainer.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Algorithms/GradientDescent/Rprop.h>
#include <shark/Algorithms/GradientDescent/SteepestDescent.h>
#include <shark/ObjectiveFunctions/NegativeGaussianProcessEvidence.h>

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_EVIDENCE
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "TestObjectiveFunction.h"


using namespace shark;
using namespace std;


BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_NegativeGaussianProcessEvidence)

BOOST_AUTO_TEST_CASE( GAUUSIAN_PROCESS_EVIDENCE )
{
	// experiment settings
	Rng::seed( 0 );
	const unsigned int ell   = 100;
	const unsigned int tests = 10000;
	const double gamma = 100.;
	const double beta  = 1000;
	const bool unconstrained = true;

	GaussianRbfKernel<> kernel(gamma, unconstrained); // parameterization for unconstraint optimization

	// loss for regression
	SquaredLoss<> loss;

	// generate dataset
	Wave prob;
	RegressionDataset trainingData = prob.generateDataset(ell);
	RegressionDataset testData = prob.generateDataset(tests);

	// define the machine
	KernelExpansion<RealVector> model;

	// define the corresponding trainer
	RegularizationNetworkTrainer<RealVector> trainer(&kernel, beta, unconstrained);

	/*
	 * Check whether evidence computations coincide.
         */
	// compute evidence
	NegativeGaussianProcessEvidence<> evidence(trainingData, &kernel, unconstrained);
	RealVector params = trainer.parameterVector();
	double prevEvidence = evidence.eval(params);

	// compute gradient
	SingleObjectiveFunction::FirstOrderDerivative derivative;
	BOOST_CHECK_SMALL(prevEvidence - evidence.evalDerivative(params, derivative), 1.e-10);
	
	/* 
	 * Check whether gradient is correct.
         */
	for(std::size_t test = 0; test != 100; ++test){
		RealVector parameters(params.size());
		for(std::size_t i = 0; i != params.size(); ++i){
			parameters(i) = Rng::uni(-2,2);
		}
		testDerivative(evidence,parameters,1.e-8);
	}

	/*
	 * Check whether optimization works.
         */
	IRpropPlus rprop;
	rprop.init(evidence, params);

	trainer.setParameterVector(rprop.solution().point);
	trainer.train(model, trainingData);
	Data<RealVector> output = model(trainingData.inputs());
	double prevTrainError = loss.eval(trainingData.labels(), output);
	output = model(testData.inputs());
	double prevTestError = loss.eval(testData.labels(), output);

	for (unsigned int iter1=0; iter1<4; iter1++) {
		for (unsigned int iter2=0; iter2<10; iter2++) 
			rprop.step(evidence);

		trainer.setParameterVector(rprop.solution().point);
		trainer.train(model, trainingData);
		output = model(trainingData.inputs());
		double trainError = loss.eval(trainingData.labels(), output);
		output = model(testData.inputs());
		double testError = loss.eval(testData.labels(), output);
		double e = rprop.solution().value;

		// we are considering the negative log evidence, so
		// both evidence as well as test error should decrease
		BOOST_CHECK(trainError < prevTrainError);
		BOOST_CHECK(e < prevEvidence);
		BOOST_CHECK(testError < prevTestError);

		prevEvidence = e;
		prevTestError = testError;
	}
}

BOOST_AUTO_TEST_SUITE_END()
