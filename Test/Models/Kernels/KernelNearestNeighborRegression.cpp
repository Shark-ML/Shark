//===========================================================================
/*!
 * 
 *
 * \brief       unit test for kernel nearest neighbor regression
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

#define BOOST_TEST_MODULE MODELS_KERNEL_NEAREST_NEIGHBOR_REGRESSION
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/NearestNeighborRegression.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Trees/KHCTree.h>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Kernels_KernelNearestNeighborRegression)

BOOST_AUTO_TEST_CASE( KERNEL_NEAREST_NEIGHBOR_REGRESSION )
{
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
	target[0](0)= -1.0;
	target[1](0)= +1.0;
	target[2](0)= -1.0;
	target[3](0)= +1.0;
	target[4](0)= -1.0;
	target[5](0)= +1.0;

	RegressionDataset dataset = createLabeledDataFromRange(input, target);

	DenseRbfKernel kernel(0.5);
	SimpleNearestNeighbors<RealVector,RealVector> algorithm(dataset, &kernel);
	NearestNeighborRegression<RealVector> model(&algorithm, 3);

	for (std::size_t i = 0; i<6; ++i)
	{
		RealVector prediction = model(input[i]);
		BOOST_CHECK_SMALL(target[i](0) - 3.0 * prediction(0), 1e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
