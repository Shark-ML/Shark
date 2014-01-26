//===========================================================================
/*!
 * 
 *
 * \brief       unit test for the leave one out error
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
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

#include <shark/ObjectiveFunctions/LooError.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>


#define BOOST_TEST_MODULE ObjectiveFunctions_LooError
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;




BOOST_AUTO_TEST_CASE( ObjectiveFunctions_LooError )
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
	inputs[4](1) = 0.5;
	std::vector<RealVector> targets(5, RealVector(1));
	targets[0](0) = 0.0;
	targets[1](0) = 0.0;
	targets[2](0) = 0.0;
	targets[3](0) = 0.0;
	targets[4](0) = 1.0;

	RegressionDataset dataset = createLabeledDataFromRange(inputs, targets);
	SquaredLoss<> loss;
	LinearModel<> model(2, 1, true);
	LinearRegression trainer;
	LooError<LinearModel<> > loo(dataset, &model, &trainer, &loss);

	// check the value of the objective function
	double value = loo.eval();
	double should = 5.0 / 9.0;                  // manually computed reference value
	BOOST_CHECK_SMALL(value - should, 1e-10);
}
