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
#ifndef SHARK_LINALG_BLAS_KERNELS_ATLAS_TRMM_HPP
#define SHARK_LINALG_BLAS_KERNELS_ATLAS_TRMM_HPP

#include "cblas_inc.hpp"
#include "../../matrix_proxy.hpp"
#include "../../vector_expression.hpp"
#include <boost/mpl/bool.hpp>

namespace shark {namespace blas {namespace bindings {

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	float const *A, int const lda,
        float* B, int const incB
) {
	cblas_strmm(order, side, uplo, transA, unit, M, N, 
		1.0,
		A, lda,
	        B, incB
	);
}

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	double const *A, int const lda,
        double* B, int const incB
) {
	cblas_dtrmm(order, side, uplo, transA, unit, M, N, 
		1.0,
		A, lda,
	        B, incB
	);
}


inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	std::complex<float> const *A, int const lda,
        std::complex<float>* B, int const incB
) {
	std::complex<float> alpha = 1.0;
	cblas_ctrmm(order, side, uplo, transA, unit, M, N, 
		static_cast<void const *>(&alpha),
		static_cast<void const *>(A), lda,
	        static_cast<void *>(B), incB
	);
}

inline void trmm(
	CBLAS_ORDER const order,
	CBLAS_SIDE const side,
	CBLAS_UPLO const uplo,
	CBLAS_TRANSPOSE const transA,
	CBLAS_DIAG const unit,
	int const M,
	int const N,
	std::complex<double> const *A, int const lda,
        std::complex<double>* B, int const incB
) {
	std::complex<double> alpha = 1.0;
	cblas_ztrmm(order, side, uplo, transA, unit, M, N, 
		static_cast<void const *>(&alpha),
		static_cast<void const *>(A), lda,
	        static_cast<void *>(B), incB
	);
}

template <bool upper, bool unit, typename MatA, typename MatB>
void trmm(
	matrix_expression<MatA> const& A,
	matrix_expression<MatB>& B,
	boost::mpl::true_
){
	SIZE_CHECK(A().size1() == A().size2());
	SIZE_CHECK(A().size2() == B().size1());
	std::size_t n = A().size1();
	std::size_t m = B().size2();
	CBLAS_DIAG cblasUnit = unit?CblasUnit:CblasNonUnit;
	CBLAS_UPLO cblasUplo = upper?CblasUpper:CblasLower;
	CBLAS_ORDER stor_ord= (CBLAS_ORDER)storage_order<typename MatA::orientation>::value;
	CBLAS_TRANSPOSE trans=CblasNoTrans;
	
	//special case: MatA and MatB do not have same storage order. in this case compute as
	//AB->B^TA^T where transpose of B is done implicitely by exchanging storage order
	CBLAS_ORDER stor_ordB= (CBLAS_ORDER)storage_order<typename MatB::orientation>::value;
	if(stor_ord != stor_ordB){
		trans = CblasTrans;
		cblasUplo=  upper?CblasLower:CblasUpper;
	}
	
	trmm(stor_ordB, CblasLeft, cblasUplo, trans, cblasUnit,
		(int)n, int(m),
	        traits::storage(A),
		traits::leading_dimension(A),
	        traits::storage(B),
	        traits::leading_dimension(B)
	);
}

template<class Storage1, class Storage2, class T1, class T2>
struct optimized_trmm_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_trmm_detail<
	dense_tag, dense_tag,
	double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trmm_detail<
	dense_tag, dense_tag,
	float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_trmm_detail<
	dense_tag, dense_tag,
	std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_trmm_detail<
	dense_tag, dense_tag,
	std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M1, class M2>
struct  has_optimized_trmm
: public optimized_trmm_detail<
	typename M1::storage_category,
	typename M2::storage_category,
	typename M1::value_type,
	typename M2::value_type
>{};

}}}
#endif
