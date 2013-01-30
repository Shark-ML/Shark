/**
 *
 * \brief Recombinates a set of individuals given a weight vector.
 * \author T.Voss
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

 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
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
