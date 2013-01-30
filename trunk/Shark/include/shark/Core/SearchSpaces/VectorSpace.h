//===========================================================================
/*!
 *  \file VectorSpace.h
 *
 *  \brief VectorSpace.h
 *
 *  \author T.Voss
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
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
