/*!
 * 
 * \file        AbstractFeasibleRegion.h
 *
 * \brief       Abstraction of the feasible region of an optimization problem.
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
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
#ifndef SHARK_CORE_ABSTRACTFEASIBLEREGION_H
#define SHARK_CORE_ABSTRACTFEASIBLEREGION_H

namespace shark {

    class Hypercube;

    /**
     * \brief Models a feasible region.
     */
    struct AbstractFeasibleRegion {

	enum Features {
	    HAS_BOUNDING_HYPERCUBE = 1
	};

	/**
	 * \brief Default c'tor.
	 */
    AbstractFeasibleRegion() : m_dimension( 0 ) {}

	/**
	 * \brief Virtual d'tor.
	 */
	virtual ~AbstractFeasibleRegion() {}

	/**
	 * \brief Accesses the dimension of the underlying search space.
	 */
	virtual std::size_t dimension() const {
	    return( m_dimension );
	}

	/**
	 * \brief Accesses the dimension of the underlying search space.
	 */
	virtual void setDimension( std::size_t dimension ) {
	    m_dimension = dimension;
	}

	const Hypercube & boundingHypercube() const;

	std::size_t m_dimension;

    };

}

#endif //ABSTRACTFEASIBLEREGION_H
