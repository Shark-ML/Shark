//===========================================================================
/*!
 *  \file KernelRegression.cpp
 *
 *  \brief test case for epsilon-SVM and regularization network
 *
 *
 *  \author T. Glasmachers
 *  \date 2011
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_KERNELREGRESSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/EpsilonSvmTrainer.h>
#include <shark/Algorithms/Trainers/RegularizationNetworkTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;


// This test case checks the resulting model of
// training eight multi-class SVMs on a minimal
// test case.
BOOST_AUTO_TEST_CASE( KERNELREGRESSION_TEST )
{
	// generate dataset
	Wave prob;
	RegressionDataset training = prob.generateDataset( 200);
	RegressionDataset test = prob.generateDataset( 10000);

	GaussianRbfKernel<> kernel(0.1);
	SquaredLoss<> loss;
	
	Data<RealVector> output;
	
	// epsilon SVM
	{
		KernelExpansion<RealVector> svm(true);
		EpsilonSvmTrainer<RealVector> trainer(&kernel, 10.0, 0.03);
		trainer.train(svm, training);
		output = svm(training.inputs());
		double train_error = loss.eval(training.labels(), output);
		BOOST_CHECK_SMALL(std::max(train_error - 0.1, 0.0), 0.01);
		output = svm(test.inputs());
		double test_error = loss.eval(test.labels(), output);
		BOOST_CHECK_SMALL(std::max(test_error - 0.1, 0.0), 0.01);
	}

	// regularization network
	{
		KernelExpansion<RealVector> svm(false);
		RegularizationNetworkTrainer<RealVector> trainer(&kernel, 0.1);
		trainer.train(svm, training);
		output = svm(training.inputs());
		double train_error = loss.eval(training.labels(), output);
		BOOST_CHECK_SMALL(std::max(train_error - 0.1, 0.0), 0.01);
		output = svm(test.inputs());
		double test_error = loss.eval(test.labels(), output);
		BOOST_CHECK_SMALL(std::max(test_error - 0.1, 0.0), 0.01);
	}
}
