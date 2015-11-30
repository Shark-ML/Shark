/*!
 * \brief       Implements the lu decomposiion algorithm
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_LINALG_BLAS_LU_HPP
#define SHARK_LINALG_BLAS_LU_HPP

#include "permutation.hpp"
#include "vector_proxy.hpp"
#include "matrix_proxy.hpp"

namespace shark {
namespace blas {

// LU factorization without pivoting
template<class M>
typename M::size_type lu_factorize(M &m) {
	typedef typename M::size_type size_type;
	typedef typename M::value_type value_type;

	size_type singular = 0;
	size_type size1 = m.size1();
	size_type size2 = m.size2();
	size_type size = (std::min)(size1, size2);
	for (size_type i = 0; i < size; ++ i) {
		matrix_column<M> mci(column(m, i));
		matrix_row<M> mri(row(m, i));
		if (m(i, i) != value_type/*zero*/()) {
			value_type m_inv = value_type(1) / m(i, i);
			subrange(mci, i + 1, size1) *= m_inv;
		} else if (singular == 0) {
			singular = i + 1;
		}
		noalias(subrange(m, i + 1, size1, i + 1, size2))-=(
		    outer_prod(subrange(mci, i + 1, size1),
		            subrange(mri, i + 1, size2)));
	}
	return singular;
}

// LU factorization with partial pivoting
template<class M, class PM>
typename M::size_type lu_factorize(M &m, PM &pm) {
	typedef typename M::size_type size_type;
	typedef typename M::value_type value_type;

	size_type singular = 0;
	size_type size1 = m.size1();
	size_type size2 = m.size2();
	size_type size = (std::min)(size1, size2);
	for (size_type i = 0; i < size; ++ i) {
		matrix_column<M> mci(column(m, i));
		matrix_row<M> mri(row(m, i));
		size_type i_norm_inf = i + index_norm_inf(subrange(mci, i, size1));
		SIZE_CHECK(i_norm_inf < size1);
		if (m(i_norm_inf, i) != value_type/*zero*/()) {
			if (i_norm_inf != i) {
				pm(i) = i_norm_inf;
				swap_rows(m,i_norm_inf,i);
			} else {
				SIZE_CHECK(pm(i) == i_norm_inf);
			}
			value_type m_inv = value_type(1) / m(i, i);
			subrange(mci, i + 1, size1) *= m_inv;
		} else if (singular == 0) {
			singular = i + 1;
		}
		noalias(subrange(m,i + 1, size1, i + 1, size2))-=(
		    outer_prod(subrange(mci, i + 1, size1),
		            subrange(mri, i + 1, size2)));
	}
	return singular;
}
}
}

#endif
