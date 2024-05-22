//===========================================================================
/*!
 *
 *
 * \brief       RemoveBudgetMaintenanceStrategy Test
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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

#define BOOST_TEST_MODULE REMOVEBUDGETMAINTENANCESTRATEGY

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/Budgeted/RemoveBudgetMaintenanceStrategy.h>
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>


using namespace shark;




BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Budgeted_RemoveBudgetMaintenanceStrategy_Test)

BOOST_AUTO_TEST_CASE( RemoveBudgetMaintenanceStrategy_reduceBudget)
{
}


BOOST_AUTO_TEST_CASE( RemoveBudgetMaintenanceStrategy_addToModel)
{
}
	

	
	
BOOST_AUTO_TEST_SUITE_END()
