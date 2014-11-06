//===========================================================================
/*!
 *
 *
 * \brief       LabelOrder Test
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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

#define BOOST_TEST_MODULE LabelOrder

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/LabelOrder.h>
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>


using namespace shark;

BOOST_AUTO_TEST_SUITE (Data_LabelOrder_Test)

BOOST_AUTO_TEST_CASE(LabelOrder_General)
{
	// create a dataset
	size_t datasetSize = 64;
	Chessboard problem(2, 0);
	LabeledData<RealVector, unsigned int> dataset = problem.generateDataset(datasetSize);

	// now map distort every, reversing is enough. make sure we have label 0 in it
	// to test if we can handle that too
	for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
		dataset.labels().element(i) = 2 * (dataset.numberOfElements() - 1) - 2 * i;

	// create a copy we can compare with later on
	LabeledData<RealVector, unsigned int> datasetCopy = dataset;

	// now reorder the dataset
	LabelOrder labelOrder;
	labelOrder.normalizeLabels(dataset);

	// obtain the ordering
	std::vector<unsigned int> internalOrder;
	labelOrder.getLabelOrder(internalOrder);

	// check the order
	for(std::size_t i = 0; i < internalOrder.size(); ++i)
		BOOST_REQUIRE_EQUAL(internalOrder[i], 2 * (dataset.numberOfElements() - 1) - 2 * i);


	// finally map the labels back on the copy
	labelOrder.restoreOriginalLabels(dataset);

	// make sure we did not loose anything
	for(std::size_t i = 0; i < internalOrder.size(); ++i)
		BOOST_REQUIRE_EQUAL(dataset.labels().element(i), datasetCopy.labels().element(i));

	// now check for some error cases:
	// create labels that are out of range and call the restore function
	LabeledData<RealVector, unsigned int> datasetBroken = dataset;
	for(std::size_t i = 0; i < dataset.numberOfElements(); ++i)
		dataset.labels().element(i) = internalOrder.size() + i;

	try
	{
		labelOrder.restoreOriginalLabels(datasetBroken);

		// this should have thrown an error.
		BOOST_REQUIRE_EQUAL(1, 0);
	}
	catch(...)
	{
		// everything is fine.
		BOOST_REQUIRE_EQUAL(0, 0);
	}
}




BOOST_AUTO_TEST_SUITE_END()
