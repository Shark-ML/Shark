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
		subrange(m, i + 1, size1, i + 1, size2).minus_assign(
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
		subrange(m,i + 1, size1, i + 1, size2).minus_assign(
		    outer_prod(subrange(mci, i + 1, size1),
		            subrange(mri, i + 1, size2)));
	}
	return singular;
}
}
}

#endif
