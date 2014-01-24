/*!
 * 
 * \file        GlobalIntermediateRecombination.h
 *
 * \brief       Recombinates a set of individuals given a weight vector.
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
#ifndef SHARK_EA_GLOBAL_INTERMEDIATE_RECOMBINATION_H
#define SHARK_EA_GLOBAL_INTERMEDIATE_RECOMBINATION_H

#include <shark/Core/Exception.h>

namespace shark {
    /**
     * \brief Recombinates a set of individuals given a weight vector.
     */
    template<typename PointType>
    struct GlobalIntermediateRecombination {

	/**
	 * \brief Carries out the recombination.
	 * 
	 * \throws shark::Exception if p.size() != weights.size().
	 */
        template<typename Population, typename Extractor>
        PointType operator()( const Population & p, const RealVector & weights, unsigned int n ) {

	    SIZE_CHECK( p.size() == weights.size() );

            PointType result( n, 0. );

            Extractor e;

            typename Population::const_iterator it;
            for( unsigned int i = 0; i < n; i++ ) {
                for( it = p.begin(); it != p.end(); ++it ) {
                    result( i ) += weights( i ) * extractor( *it )( i );
                }
            }
            return( result );
        }
    };
}

#endif
