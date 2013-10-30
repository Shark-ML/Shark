//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2010
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
//===========================================================================
#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_gemv_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_ATLAS_gemv_HPP

#include "cblas_inc.hpp"

namespace shark {namespace blas {namespace bindings {

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, float const *A, int const lda,
        float const *X, int const incX,
        double beta, float *Y, int const incY
) {
	cblas_sgemv(Order, TransA, M, N, alpha, A, lda,
	        X, incX,
	        beta, Y, incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha, double const *A, int const lda,
        double const *X, int const incX,
        double beta, double *Y, int const incY
) {
	cblas_dgemv(Order, TransA, M, N, alpha, A, lda,
	        X, incX,
	        beta, Y, incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
        double alpha,
        std::complex<float> const *A, int const lda,
        std::complex<float> const *X, int const incX,
        double beta,
        std::complex<float> *Y, int const incY
) {
	std::complex<float> alphaArg(alpha,0);
	std::complex<float> betaArg(beta,0);
	cblas_cgemv(Order, TransA, M, N,
	        static_cast<void const *>(&alphaArg),
	        static_cast<void const *>(A), lda,
	        static_cast<void const *>(X), incX,
	        static_cast<void const *>(&betaArg),
	        static_cast<void *>(Y), incY);
}

inline void gemv(CBLAS_ORDER const Order,
        CBLAS_TRANSPOSE const TransA, int const M, int const N,
         double alpha,
        std::complex<double> const *A, int const lda,
        std::complex<double> const *X, int const incX,
        double beta,
        std::complex<double> *Y, int const incY
) {
	std::complex<double> alphaArg(alpha,0);
	std::complex<double> betaArg(beta,0);
	cblas_zgemv(Order, TransA, M, N,
	        static_cast<void const *>(&alphaArg),
	        static_cast<void const *>(A), lda,
	        static_cast<void const *>(X), incX,
	        static_cast<void const *>(&betaArg),
	        static_cast<void *>(Y), incY);
}


// y <- alpha * op (A) * x + beta * y
// op (A) == A || A^T || A^H
template <typename MatrA, typename VectorX, typename VectorY>
void gemv(
	 matrix_expression<MatrA> const &A,
	vector_expression<VectorX> const &x,
        vector_expression<VectorY> &y,
	typename VectorY::value_type alpha,
	boost::mpl::true_
){
	std::size_t m = A().size1();
	std::size_t n = A().size2();
	
	SIZE_CHECK(x().size() == A().size2());
	SIZE_CHECK(y().size() == A().size1());

	CBLAS_ORDER const stor_ord= (CBLAS_ORDER)storage_order<typename MatrA::orientation>::value;
	
	gemv(stor_ord, CblasNoTrans, (int)m, (int)n, alpha,
	        traits::storage(A),
		traits::leading_dimension(A),
	        traits::storage(x),
	        traits::stride(x),
	        typename VectorY::value_type(1),
	        traits::storage(y),
	        traits::stride(y)
	);
}

template<class Storage1, class Storage2, class Storage3, class T1, class T2, class T3>
struct optimized_gemv_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_gemv_detail<
	dense_tag, dense_tag, dense_tag, 
	double, double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_gemv_detail<
	dense_tag, dense_tag, dense_tag, 
	float, float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_gemv_detail<
	dense_tag, dense_tag, dense_tag,
	std::complex<double>, std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_gemv_detail<
	dense_tag, dense_tag, dense_tag,
	std::complex<float>, std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class M1, class M2, class M3>
struct  has_optimized_gemv
: public optimized_gemv_detail<
	typename M1::storage_category,
	typename M2::storage_category,
	typename M3::storage_category,
	typename M1::value_type,
	typename M2::value_type,
	typename M3::value_type
>{};

}}}
#endif
