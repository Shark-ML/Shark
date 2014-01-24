/*!
 * 
 * \file        FitnessTraits.h
 *
 * \brief       Models extraction of fitness values.
 * 
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_EA_FITNESS_TRAITS_H
#define SHARK_EA_FITNESS_TRAITS_H

#include <shark/Algorithms/DirectSearch/EA.h>

#include <vector>

namespace shark {

    /**
     * \brief Abstracts extraction of fitness values from individuals.
     */
    template<typename T>
	struct FitnessTraits {
	    
	    /**
	     * \brief Returns a const reference to penalized fitness values.
	     */
	    std::vector<double> & operator()( T & t, tag::PenalizedFitness ) const {
		return( t );
	    }

	    /**
	     * \brief Returns a mutable reference penalized fitness values.
	     */
	    const std::vector<double> & operator()( const T & t, tag::PenalizedFitness ) const {
		return( t );
	    }

	    /**
	     * \brief Returns a const reference to unpenalized fitness values.
	     */
	    std::vector<double> & operator()( T & t, tag::UnpenalizedFitness ) const {
		return( t );
	    }

	    /**
	     * \brief Returns a mutable reference to unpenalized fitness values.
	     */
	    const std::vector<double> & operator()( const T & t, tag::UnpenalizedFitness ) const {
		return( t );
	    }
	};

}

#endif
