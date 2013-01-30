/**
 *
 * \brief Convex quadratic benchmark function.
 * \author T. Voss
 * \date 2010-2011
 *
 * \par Copyright (c):
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SPHERE_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SPHERE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
    /**
     * \brief Convex quadratic benchmark function.
     */
    struct Sphere : public AbstractObjectiveFunction< VectorSpace<double>,double > {
	typedef AbstractObjectiveFunction<VectorSpace<double>,double> super;

	Sphere(unsigned int numberOfVariables = 5) {
	    m_features |= CAN_PROPOSE_STARTING_POINT;
	    m_name = "Sphere";
	}

	void configure( const PropertyTree & node ) {
	    m_numberOfVariables = node.get( "numberOfVariables", 5l );
	}

	void proposeStartingPoint( super::SearchPointType & x ) const {
	    x.resize( m_numberOfVariables );

	    for( unsigned int i = 0; i < x.size(); i++ ) {
		x( i ) = Rng::uni( 0, 1 );
	    }
	}

	double eval( const super::SearchPointType & p ) const {
	    m_evaluationCounter++;
	    double sum = 0;
	    for( unsigned int i = 0; i < p.size(); i++ )
		sum += sqr( p( i ) );

	    return sum;
	}
    };

    ANNOUNCE_SINGLE_OBJECTIVE_FUNCTION( Sphere, shark::soo::RealValuedObjectiveFunctionFactory );
}

#endif
