//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_KERNELS_ATLAS_TRMV_HPP
#define SHARK_LINALG_BLAS_KERNELS_ATLAS_TRMV_HPP

#include "cblas_inc.hpp"
#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark {namespace blas {namespace bindings {

inline void trmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	float const *A, int const lda,
        float* X, int const incX
) {
	cblas_strmv(Order, uplo, transA, unit, N, 
		A, lda,
	        X, incX
	);
}

inline void trmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	double const *A, int const lda,
        double* X, int const incX
) {
	cblas_dtrmv(Order, uplo, transA, unit, N, 
		A, lda,
	        X, incX
	);
}


inline void trmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	std::complex<float> const *A, int const lda,
        std::complex<float>* X, int const incX
) {
	cblas_ctrmv(Order, uplo, transA, unit, N, 
		static_cast<void const *>(A), lda,
	        static_cast<void *>(X), incX
	);
}

inline void trmv(
	CBLAS_ORDER const Order,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const N,
	std::complex<double> const *A, int const lda,
        std::complex<double>* X, int const incX
) {
	cblas_ztrmv(Order, uplo, transA, unit, N, 
		static_cast<void const *>(A), lda,
	        static_cast<void *>(X), incX
	);
}

template <bool upper, bool unit, typename MatrA, typename VectorX>
void trmv(
	matrix_expression<MatrA> const& A,
	vector_expression<VectorX> &x,
	boost::mpl::true_
){
	SIZE_CHECK(x().size() == A().size2());
	SIZE_CHECK(A().size2() == A().size1());
	std::size_t n = A().size1();
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_UPLO cblasUplo = upper?CblasUpper:CblasLower;
	CBLAS_ORDER stor_ord= (CBLAS_ORDER)storage_order<typename MatrA::orientation>::value;
	
	trmv(stor_ord, cblasUplo, CblasNoTrans, cblasUnit, (int)n,
	        traits::storage(A),
		traits::leading_dimension(A),
	        traits::storage(x),
	        traits::stride(x)
	);
}

template<class Storage1, class Storage2, class T1, class T2>
struct optimized_trmv_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_trmv_detail<
	dense_tag, dense_tag,
	double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trmv_detail<
	dense_tag, dense_tag,
	float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_trmv_detail<
	dense_tag, dense_tag,
	std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trmv_detail<
	dense_tag, dense_tag,
	std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M1, class M2>
struct  has_optimized_trmv
: public optimized_trmv_detail<
	typename M1::storage_category,
	typename M2::storage_category,
	typename M1::value_type,
	typename M2::value_type
>{};

}}}
#endif
