//===========================================================================
/*!
 * 
 *
 * \brief       Absolute loss test case
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

#include <shark/ObjectiveFunctions/Loss/AbsoluteLoss.h>
#include <cmath>

#include <shark/ObjectiveFunctions/ErrorFunction.h>
#include <shark/Models/LinearModel.h>
#include <shark/Rng/GlobalRng.h>

#define BOOST_TEST_MODULE OBJECTIVEFUNCTIONS_ABSOLUTELOSS
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

using namespace shark;

BOOST_AUTO_TEST_SUITE (ObjectiveFunctions_AbsoluteLoss)

BOOST_AUTO_TEST_CASE( ABSOLUTELOSS_EVAL ) {
	AbsoluteLoss<> loss;
	unsigned int maxTests = 1000;
	unsigned int batchSize = 10;
	for (unsigned int test = 0; test != maxTests; ++test)
	{
		std::size_t dim = Rng::discrete(5, 100);
		RealMatrix target(batchSize,dim);
		RealMatrix output(batchSize,dim);
		double sum = 0;
		for(std::size_t b = 0; b != batchSize; ++b){
			double dist2 = 0;
			for (std::size_t d=0; d != dim; d++){
				target(b,d) = Rng::gauss();
				output(b,d) = Rng::gauss();
				double diff = target(b,d) - output(b,d);
				dist2 += diff * diff;
			}
			sum += std::sqrt(dist2);
		}
		
		double l = loss.eval(target, output);
		BOOST_CHECK_SMALL(l - sum, 1e-12);
	}
}

BOOST_AUTO_TEST_SUITE_END()
