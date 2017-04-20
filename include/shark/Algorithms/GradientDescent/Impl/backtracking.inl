//===========================================================================
/*!
 *  \brief 
 *  \author O. Krause
 *  \date   2017
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_ALGORITHMS_GRADIENTDESCENT_IMPL_BACKTRACKING_INL
#define SHARK_ALGORITHMS_GRADIENTDESCENT_IMPL_BACKTRACKING_INL

namespace shark {

namespace detail{
/// \brief backtracking line search statisfying the weak wolfe conditions
template <class VectorT, class Function>
void backtracking(
	VectorT &point,
	const VectorT &searchDirection,
	double &value,
	Function const& func,
	VectorT &gradient,
	double t = 1.0
) {
	SIZE_CHECK(point.size() == searchDirection.size());
	SIZE_CHECK(point.size() == gradient.size());

	// Constants
	const std::size_t maxIter = 50; //maximum number of iterations to try
	const double shrinking = 0.7;//shrinking factor when condition is not yet fulfilled
	const double c1 = 1e-4;//constant for weak wolfe condition
	
	double gtd = inner_prod(gradient, searchDirection);
	// Initial step values
	VectorT g_new(point.size());
	double f_new = value;
	
	std::size_t iter = 0;
	while(iter < maxIter) {
		double f_new  = func.evalDerivative(point + t * searchDirection, g_new);
		if (f_new < value + c1 * t * gtd) {
			break;
		}else{
			t *= shrinking;
			++iter;
		}
	}

	if (iter < maxIter){
		noalias(point) += t * searchDirection;
		value = f_new;
		gradient = g_new;
	}
}

}}


#endif
