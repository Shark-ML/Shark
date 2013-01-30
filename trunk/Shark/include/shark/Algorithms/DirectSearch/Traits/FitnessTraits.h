/**
 *
 * \brief Models extraction of fitness values.
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
