//
//  Copyright (c) 2000-2002
//  Joerg Walter, Mathias Koch
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_LU_
#define _BOOST_UBLAS_LU_

#include <shark/LinAlg/BLAS/ublas/operation.hpp>
#include <shark/LinAlg/BLAS/ublas/vector_proxy.hpp>
#include <shark/LinAlg/BLAS/ublas/matrix_proxy.hpp>
#include <shark/LinAlg/BLAS/ublas/vector.hpp>
#include <shark/LinAlg/BLAS/ublas/triangular.hpp>

// LU factorizations in the spirit of LAPACK and Golub & van Loan

namespace shark {
namespace blas {

/** \brief Permutation matrix
 *
 * \tparam T
 * \tparam A
 */
template<class T = std::size_t, class A = unbounded_array<T> >
class permutation_matrix:
	public vector<T, A> {
public:
	typedef vector<T, A> vector_type;
	typedef typename vector_type::size_type size_type;

	// Construction and destruction

	explicit
	permutation_matrix(size_type size):
		vector<T, A> (size) {
		for (size_type i = 0; i < size; ++ i)
			(*this)(i) = i;
	}

	explicit
	permutation_matrix(const vector_type &init)
		: vector_type(init)
	{ }

	~permutation_matrix() {}

	// Assignment

	permutation_matrix &operator = (const permutation_matrix &m) {
		vector_type::operator = (m);
		return *this;
	}
};

template<class PM, class MV>
void swap_rows(const PM &pm, blas::vector_expression<MV>& mv) {
	typedef typename PM::size_type size_type;

	size_type size = pm.size();
	for (size_type i = 0; i < size; ++ i) {
		if (i != pm(i))
			std::swap(mv()(i), mv()(pm(i)));
	}
}
template<class PM, class MV>
void swap_rows(const PM &pm, blas::matrix_expression<MV>& mv) {
	typedef typename PM::size_type size_type;

	size_type size = pm.size();
	for (size_type i = 0; i < size; ++ i) {
		if (i != pm(i))
			row(mv(), i).swap(row(mv(), pm(i)));
	}
}

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
		size_type i_norm_inf = i + index_norm_inf(subrange(mci, range(i, size1)));
		BOOST_UBLAS_CHECK(i_norm_inf < size1, external_logic());
		if (m(i_norm_inf, i) != value_type/*zero*/()) {
			if (i_norm_inf != i) {
				pm(i) = i_norm_inf;
				row(m, i_norm_inf).swap(mri);
			} else {
				BOOST_UBLAS_CHECK(pm(i) == i_norm_inf, external_logic());
			}
			value_type m_inv = value_type(1) / m(i, i);
			subrange(mci, range(i + 1, size1)) *= m_inv;
		} else if (singular == 0) {
			singular = i + 1;
		}
		subrange(m, range(i + 1, size1), range(i + 1, size2)).minus_assign(
		    outer_prod(subrange(mci, range(i + 1, size1)),
		            subrange(mri, range(i + 1, size2))));
	}
	return singular;
}


// LU substitution
template<class M, class E>
void lu_substitute(const M &m, vector_expression<E> &e) {
	inplace_solve(m, e, unit_lower_tag());
	inplace_solve(m, e, upper_tag());
}
template<class M, class E>
void lu_substitute(const M &m, matrix_expression<E> &e) {
	inplace_solve(m, e, unit_lower_tag());
	inplace_solve(m, e, upper_tag());
}
template<class M, class PMT, class PMA, class MV>
void lu_substitute(const M &m, const permutation_matrix<PMT, PMA> &pm, MV &mv) {
	swap_rows(pm, mv);
	lu_substitute(m, mv);
}
template<class E, class M>
void lu_substitute(vector_expression<E> &e, const M &m) {
	inplace_solve(e, m, upper_tag());
	inplace_solve(e, m, unit_lower_tag());
}
template<class E, class M>
void lu_substitute(matrix_expression<E> &e, const M &m) {
	inplace_solve(e, m, upper_tag());
	inplace_solve(e, m, unit_lower_tag());
}
template<class MV, class M, class PMT, class PMA>
void lu_substitute(MV &mv, const M &m, const permutation_matrix<PMT, PMA> &pm) {
	swap_rows(pm, mv);
	lu_substitute(mv, m);
}

}
}

#endif
