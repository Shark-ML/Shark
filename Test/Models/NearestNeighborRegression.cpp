//===========================================================================
/*!
 * 
 *
 * \brief       Unit test for nearest neighbor regression.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2012
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
//===========================================================================
#define BOOST_TEST_MODULE Models_NearestNeighborRegression
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/NearestNeighborRegression.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_NearestNeighborRegression)

BOOST_AUTO_TEST_CASE( Models_NearestNeighborRegression ) {

	// simple data set with paired points
	std::vector<RealVector> input(6, RealVector(2));
	input[0](0)=1;
	input[0](1)=3;
	input[1](0)=-1;
	input[1](1)=3;
	input[2](0)=1;
	input[2](1)=0;
	input[3](0)=-1;
	input[3](1)=0;
	input[4](0)=1;
	input[4](1)=-3;
	input[5](0)=-1;
	input[5](1)=-3;
	std::vector<RealVector> target(6, RealVector(1));
	target[0](0)=-5.0;
	target[1](0)=-3.0;
	target[2](0)=-1.0;
	target[3](0)=+1.0;
	target[4](0)=+3.0;
	target[5](0)=+5.0;
	RegressionDataset dataset = createLabeledDataFromRange(input, target);

	// model
	DenseLinearKernel kernel;
	SimpleNearestNeighbors<RealVector,RealVector> algorithm(dataset, &kernel);
	NearestNeighborRegression<RealVector> model(&algorithm, 2);

	// predictions must be pair averages
	Data<RealVector> prediction = model(dataset.inputs());
	for (int i = 0; i<6; ++i)
	{
		BOOST_CHECK_SMALL(prediction.element(i)(0) - 4.0 * (i/2 - 1), 1e-14);
	}
}

BOOST_AUTO_TEST_SUITE_END()
