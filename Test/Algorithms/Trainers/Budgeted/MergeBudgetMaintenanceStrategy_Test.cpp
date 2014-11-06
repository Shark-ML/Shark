//===========================================================================
/*!
 *
 *
 * \brief       MergeBudgetMaintenanceStrategy Test
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

#define BOOST_TEST_MODULE MERGEBUDGETMAINTENANCESTRATEGY

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Algorithms/Trainers/Budgeted/MergeBudgetMaintenanceStrategy.h>
#include <shark/Data/DataDistribution.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/KernelExpansion.h>


using namespace shark;


BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_Budgeted_MergeBudgetMaintenanceStrategy_Test)

BOOST_AUTO_TEST_CASE( MergeBudgetMaintenanceStrategy_MergingProblemFunction)
{
    // setup
    MergeBudgetMaintenanceStrategy<RealVector> ms;
    RealVector h(1);     // initial search starting point
    RealVector xi(1);    // direction of search
    h(0) = 0.0;
    xi(0) = 0.5;

    // test for the minimum of the easy function  1 + 1=2=const between 0 and 1
    double fret(0.5);
    double k = 1.0;
    double a = 1.0;
    double b = 1.0;
    {
        MergeBudgetMaintenanceStrategy<RealVector>::MergingProblemFunction mergingProblemFunction(a, b, k);
        detail::dlinmin(h, xi, fret, mergingProblemFunction, 0.0, 1.0);
        BOOST_REQUIRE_EQUAL(h(0), 0);
    }

    // test for the minimum of the easy function  0.5^{h*h} between 0 and 1
    fret = 0.5;
    k = 0.5;
    a = 0.0;
    b = 1.0;
    h(0) = 0.0;
    xi(0) = 0.00001;
    {
        MergeBudgetMaintenanceStrategy<RealVector>::MergingProblemFunction mergingProblemFunction(a, b, k);
        detail::dlinmin(h, xi, fret, mergingProblemFunction, 0.0, 1.0);
        BOOST_REQUIRE_SMALL(h(0), 0.000001);
    }
    
    // minimize ( -0.2*(0.2)^{x*x} - 0.1* (0.2)^{ (1-x) * (1-x)})  over [0,1]
    fret = 0.0;
    k = 0.2;
    a = 0.1;
    b = 0.2;
    h(0) = 0.0;
    xi(0) = 0.00001;
    {
        MergeBudgetMaintenanceStrategy<RealVector>::MergingProblemFunction mergingProblemFunction(a, b, k);
        detail::dlinmin(h, xi, fret, mergingProblemFunction, 0.0, 1.0);
        BOOST_REQUIRE_SMALL(h(0) -0.133040685 , 0.000001);
    }
}



BOOST_AUTO_TEST_CASE( MergeBudgetMaintenanceStrategy_reduceBudget)
{
}


BOOST_AUTO_TEST_CASE( MergeBudgetMaintenanceStrategy_addToModel)
{
}
	

	
	
BOOST_AUTO_TEST_SUITE_END()
