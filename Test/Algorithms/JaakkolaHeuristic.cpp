//===========================================================================
/*!
 * 
 *
 * \brief       Unit test for Jaakkola's bandwidth selection heuristic.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2012
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

#include <shark/Algorithms/JaakkolaHeuristic.h>

#define BOOST_TEST_MODULE Algorithms_JaakkolaHeuristic
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;


BOOST_AUTO_TEST_CASE( Algorithms_JaakkolaHeuristic )
{
	// Create a simple data set.
	// The distribution of distances of pairs of
	// different classes is: 1 2 2 3 3 3 4 4 5.
	// We have: minimum=1, median=3, maximum=5.
	std::vector<RealVector> inputs(6, RealVector(1));
	inputs[0](0) = 0.0;
	inputs[1](0) = 1.0;
	inputs[2](0) = 2.0;
	inputs[3](0) = 3.0;
	inputs[4](0) = 4.0;
	inputs[5](0) = 5.0;
	std::vector<unsigned int> targets(6);
	targets[0] = 0;
	targets[1] = 0;
	targets[2] = 0;
	targets[3] = 1;
	targets[4] = 1;
	targets[5] = 1;
	ClassificationDataset dataset = createLabeledDataFromRange(inputs, targets);

	// todo: make test for default version
	//
	// obtain values of sigma for different quantiles, including the default
	JaakkolaHeuristic jh(dataset, false);
	double sigma         = jh.sigma();
	double sigma_minimum = jh.sigma(0.0);
	double sigma_median  = jh.sigma(0.5);
	double sigma_maximum = jh.sigma(1.0);
	double gamma_median  = jh.gamma(0.5);

	// check values
	BOOST_CHECK_SMALL(std::abs(sigma_minimum - 1.0), 1e-14);
	BOOST_CHECK_SMALL(std::abs(sigma_median  - 3.0), 1e-14);
	BOOST_CHECK_SMALL(std::abs(sigma_maximum - 5.0), 1e-14);

	// check consistency
	BOOST_CHECK_SMALL(std::abs(sigma_median - sigma), 1e-14);
	BOOST_CHECK_SMALL(std::abs(gamma_median - 0.5 / (sigma_median * sigma_median)), 1e-14);
}
