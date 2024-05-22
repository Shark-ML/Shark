//===========================================================================
/*!
 * 
 *
 * \brief       test case for regularization network
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2013
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#define BOOST_TEST_MODULE ALGORITHMS_TRAINERS_REGULARIZATION_NETWORK
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/RegularizationNetworkTrainer.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataDistribution.h>


using namespace shark;


BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_RegularizationNetworkTrainer)

BOOST_AUTO_TEST_CASE( REGULARIZATION_NETWORK_TEST )
{
	const std::size_t ell = 200;
	const double lambda = 1e-6;
	const double threshold = 1e-8;

	Wave prob(0.0, 5.0);
	RegressionDataset training = prob.generateDataset(ell);

	GaussianRbfKernel<> kernel(0.1);
	KernelExpansion<RealVector> svm;
	RegularizationNetworkTrainer<RealVector> trainer(&kernel, lambda);
	trainer.train(svm, training);

	Data<RealVector> output;
	output = svm(training.inputs());

	RealVector alpha = svm.parameterVector();
	for (std::size_t i=0; i<training.numberOfElements(); i++)
	{
		double y = training.labels().element(i)(0);
		double f = output.element(i)(0);
		double xi = (f - y) * (f - y);
		BOOST_CHECK_SMALL(xi, threshold);
	}
}

BOOST_AUTO_TEST_SUITE_END()
