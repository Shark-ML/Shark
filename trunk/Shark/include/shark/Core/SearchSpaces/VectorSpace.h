//===========================================================================
/*!
 * 
 *
 * \brief       VectorSpace.h
 * 
 * 
 *
 * \author      T.Voss
 * \date        2010
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

#ifndef SHARK_CORE_VECTORSPACE_H
#define SHARK_CORE_VECTORSPACE_H

#include <shark/LinAlg/Base.h>

namespace shark {

	/**
	* \brief Models the concept of a vector space over.
	* \tparam Scalar The underlying field.
	*/
    template<typename Scalar>
	struct VectorSpace {

		/**
		* \brief Tags the search space as a vector space.
		*/
		BOOST_STATIC_CONSTANT( bool, IS_VECTOR_SPACE = true );

		/**
		* \brief Defines the elements of the vector space over the supplied field.
		*/
	    typedef blas::vector< Scalar > PointType;
	};

};

#endif
