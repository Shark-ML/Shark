/*!
 * \brief       kernels for folding kernels with openCL
 * 
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_LINALG_BLAS_KERNELS_CLBLAS_VECTOR_FOLD_HPP
#define SHARK_LINALG_BLAS_KERNELS_CLBLAS_VECTOR_FOLD_HPP

#include "../../expression_types.hpp"
#include <boost/compute/container/array.hpp>
#include <boost/compute/algorithm/reduce.hpp>

namespace shark{namespace blas {namespace bindings{

template<class F, class V>
void vector_fold(vector_expression<V, gpu_tag> const& v, typename F::result_type& value, dense_tag) { 
	boost::compute::array<typename F::result_type,1> val;
	val[0] = value;
	boost::compute::reduce(v().begin(), v().end(),  val.begin(), F(), v().queue());
	value = val[0];
}


}}}
#endif
