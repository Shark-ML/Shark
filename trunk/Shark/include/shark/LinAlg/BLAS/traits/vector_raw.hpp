//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2012
 *
 *  \par Copyright (c) 1998-2011:
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

#ifndef SHARK_LINALG_BLAS_TRAITS_VECTOR_RAW_HPP
#define SHARK_LINALG_BLAS_TRAITS_VECTOR_RAW_HPP

#include "metafunctions.h"


namespace shark { namespace blas{ namespace traits {
	
template <typename V>
int vector_stride(blas::vector_expression<V> const&v) { 
	return ExpressionTraits<V const>::stride(v());
}

template <typename V>
typename PointerType<V const>::type vector_storage(blas::vector_expression<V> const& v) { 
	return ExpressionTraits<V const>::storageBegin(v());
}
template <typename V>
typename PointerType<V>::type vector_storage(blas::vector_expression<V>& v) { 
	return ExpressionTraits<V>::storageBegin(v());
}

template <typename V>
bool isSparse(blas::vector_expression<V> const& ) { 
	return IsSparse<V>::value;
}

}}}

#endif
