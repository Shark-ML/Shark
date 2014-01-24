//===========================================================================
/*!
 * 
 * \file        Derivative.h
 *
 * \brief       wrapper for first and second derivative
 * 
 * 
 *
 * \author      T.Voss, T
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
//===========================================================================
#ifndef SHARK_CORE_DERIVATIVE_H
#define SHARK_CORE_DERIVATIVE_H

namespace shark {
// namespace detail {


/// \brief Encapsulation of the first order derivative information, consisting of the gradient only.
template<typename VectorT>
struct TypedFirstOrderDerivative {
	typedef VectorT VectorType;

	VectorType m_gradient;
};

/// \brief Encapsulation of the second order derivative information, consisting of the gradient and the Hessian.
template<typename VectorT, typename MatrixT>
struct TypedSecondOrderDerivative {
	typedef VectorT VectorType;
	typedef MatrixT MatrixType;

	VectorType m_gradient;
	MatrixType m_hessian;
};


}
#endif // SHARK_CORE_DERIVATIVE_H
