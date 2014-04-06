/*!
 * 
 *
 * \brief       Two-dimensional, real-valued Himmelblau function.
 * 
 * Multi-modal benchmark function.
 * 
 * 
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_HIMMELBLAU_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_HIMMELBLAU_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/** 
* \brief Multi-modal two-dimensional continuous Himmelblau benchmark function.
*
* Implements Himmelblau's real-valued, multi-modal benchmark function. The
* function is limited to two dimensions. Please see:
*   http://en.wikipedia.org/wiki/Himmelblau%27s_function
* for further information.
*/
struct Himmelblau : public SingleObjectiveFunction {
	/**
	* \brief Constructs an instance of the function.
	*/
	Himmelblau() {
		m_features|=CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Himmelblau"; }

	std::size_t numberOfVariables()const{
		return 2;
	}

	void configure( const PropertyTree & node ) {
		(void) node;
	}

	void proposeStartingPoint( SearchPointType & x ) const {
		x.resize( 2 );

		for( unsigned int i = 0; i < x.size(); i++ ) {
			x( i ) = Rng::uni( -3, 3 );
		}
	}

	/**
	* \brief Evaluates the function for the supplied search point.
	* \throws shark::Exception if the size of p does not equal 2.
	*/
	double eval( const SearchPointType & p ) const {
		SIZE_CHECK(p.size() == 2);

		m_evaluationCounter++;

		return( 
			sqr( sqr( p( 0 ) ) + p( 1 ) - 11 ) +
			sqr( p( 0 ) + sqr( p( 1 ) ) - 7 )
		);
	}
};

}

#endif
