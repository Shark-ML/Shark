//===========================================================================
/*!
 * 
 *
 * \brief       test case for epsilon-SVM
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2013
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_EPSILON_SVM
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/EpsilonSvmTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;


BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_EpsilonSvmTrainer)

BOOST_AUTO_TEST_CASE( EPSILON_SVM_TEST )
{
	const std::size_t ell = 200;
	const double C = 10.0;
	const double epsilon = 0.03;
	const double accuracy = 1e-8;

	Wave prob;
	RegressionDataset training = prob.generateDataset(ell);

	GaussianRbfKernel<> kernel(0.1);
	KernelExpansion<RealVector> svm;
	EpsilonSvmTrainer<RealVector> trainer(&kernel, C, epsilon);
	trainer.stoppingCondition().minAccuracy = accuracy;
	trainer.sparsify() = false;
	trainer.train(svm, training);

	Data<RealVector> output;
	output = svm(training.inputs());

	RealVector alpha = svm.parameterVector();
	for (std::size_t i=0; i<training.numberOfElements(); i++)
	{
		double y = training.labels().element(i)(0);
		double f = output.element(i)(0);
		double a = alpha(i);
		double xi = std::max(0.0, std::abs(f - y) - epsilon);
		if (a == 0.0)
		{
			BOOST_CHECK_SMALL(xi, accuracy);
		}
		else
		{
			BOOST_CHECK(xi >= -accuracy);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END()
