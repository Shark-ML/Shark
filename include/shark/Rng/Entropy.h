/*!
 * 
 *
 * \brief       Provides a function for estimating the entropy of a distribution.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010-01-01
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
#ifndef SHARK_RNG_ENTROPY_H
#define SHARK_RNG_ENTROPY_H

#include <shark/Core/Exception.h>

namespace shark {

	/**
	* \brief Estimates the entropy of a distribution. The more trials the better is the estimate, but good estimates are slow.
	*
	* \tparam Distribution The type of distribution, needs to provide operator()() for sampling purposes.
	* \param [in,out] d Distribution instance to sample from.
	* \param [in] trials The number of samples drawn from the distribution. Needs to be larger than 0.
	* \throws shark::Exception if trials == 0.
	*/
	template<typename Distribution>
	double entropy( Distribution & d, std::size_t trials = 10000 ) {
		if( trials == 0 )
			throw( shark::Exception( "entropy: Trial count needs to be larger than 0.", __FILE__, __LINE__ ) );

		double t = 0;
		for ( unsigned int i = 0; i < trials; i++ ) {
			t += ::log( d.p( d() ) );
		}
		return( -t / trials );
	}

}

#endif