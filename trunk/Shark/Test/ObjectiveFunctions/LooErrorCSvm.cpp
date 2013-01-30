//===========================================================================
/*!
 *
 *  \brief Unit test for the leave one out error for C-SVMs.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2011
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

#include <shark/ObjectiveFunctions/LooErrorCSvm.h>
#include <shark/ObjectiveFunctions/LooError.h>
#include <shark/Models/Kernels/LinearKernel.h>


#define BOOST_TEST_MODULE ObjectiveFunctions_LooErrorCSvm
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;




BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooErrorCSvm )
{
	std::vector<RealVector> inputs(5, RealVector(2));
	inputs[0](0) = 0.0;
	inputs[0](1) = 0.0;
	inputs[1](0) = 1.0;
	inputs[1](1) = 0.0;
	inputs[2](0) = 0.0;
	inputs[2](1) = 1.0;
	inputs[3](0) = 1.0;
	inputs[3](1) = 1.0;
	inputs[4](0) = 0.5;
	inputs[4](1) = 0.4;
	std::vector<unsigned int> targets(5);
	targets[0] = 0;
	targets[1] = 0;
	targets[2] = 1;
	targets[3] = 1;
	targets[4] = 1;
	ClassificationDataset dataset(inputs, targets);

	// SVM setup
	LinearKernel<> kernel;
	double C = 1e100;  // hard margin
	CSvmTrainer<RealVector> trainer(&kernel, C);

	// efficiently computed loo error
	LooErrorCSvm<RealVector> loosvm(dataset, &trainer, true);
	double value = loosvm.eval();

	// brute force computation
	ZeroOneLoss<unsigned int, RealVector> loss;
	KernelExpansion<RealVector> ke(&kernel, true);
	LooError<KernelExpansion<RealVector>,unsigned int> loo(dataset, &ke, &trainer, &loss);
	double should = loo.eval();

	// compare to brute force computation
	// and to geometric intuition (3 out of
	// 5 are errors)
	BOOST_CHECK_SMALL(value - should, 1e-10);
	BOOST_CHECK_SMALL(value - 0.6, 1e-10);
}
