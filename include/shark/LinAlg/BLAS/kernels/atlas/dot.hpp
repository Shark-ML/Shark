//===========================================================================
/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_LINALG_BLAS_KERNELS_ATLAS_DOT_HPP
#define SHARK_LINALG_BLAS_KERNELS_ATLAS_DOT_HPP

#include "cblas_inc.hpp"

namespace shark {namespace blas {namespace bindings {

inline void dot(int N, 
	float const* x, int strideX,
	float const* y, int strideY,
	float&  result
) {
	result  = cblas_sdot(N, x, strideX, y, strideY);
}

inline void dot(int N, 
	double const* x, int strideX,
	double const* y, int strideY,
	double&  result
) {
	result  = cblas_ddot(N, x, strideX, y, strideY);
}

inline void dot(int N, 
	std::complex<float> const* x, int strideX,
	std::complex<float> const* y, int strideY,
	std::complex<float>& result
) {
	cblas_cdotu_sub(N, 
		static_cast<void const* >(x), strideX, 
		static_cast<void const* >(y), strideY,
		static_cast<void*>(&result)
	);
}

inline void dot(int N, 
	std::complex<double> const* x, int strideX,
	std::complex<double> const* y, int strideY,
	std::complex<double>& result
) {
	cblas_zdotu_sub(N, 
		static_cast<void const* >(x), strideX,
		static_cast<void const* >(y), strideY,
		static_cast<void*>(&result)
	);
}


// y <- alpha*  op (A)*  x + beta*  y
// op (A) == A || A^T || A^H
template <typename VectorX, typename VectorY>
void dot(
	vector_expression<VectorX> const&   x,
        vector_expression<VectorY> const& y,
	typename VectorX::value_type& result,
	boost::mpl::true_
){
	SIZE_CHECK(x().size() == y().size());

	dot(
		x().size(),
		traits::storage(x), traits::stride(x),
		traits::storage(y), traits::stride(y),
		result
	);
}

template<class Storage1, class Storage2, class T1, class T2, class T3>
struct optimized_dot_detail{
	typedef boost::mpl::false_ type;
};
template<>
struct optimized_dot_detail<
	dense_tag, dense_tag,
	double, double, double
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_dot_detail<
	dense_tag, dense_tag,
	float, float, float
>{
	typedef boost::mpl::true_ type;
};

template<>
struct optimized_dot_detail<
	dense_tag, dense_tag,
	std::complex<double>, std::complex<double>, std::complex<double>
>{
	typedef boost::mpl::true_ type;
};
template<>
struct optimized_dot_detail<
	dense_tag, dense_tag,
	std::complex<float>, std::complex<float>, std::complex<float>
>{
	typedef boost::mpl::true_ type;
};

template<class V1, class V2, class result_type>
struct  has_optimized_dot
: public optimized_dot_detail<
	typename V1::storage_category,
	typename V2::storage_category,
	typename V1::value_type,
	typename V2::value_type,
	result_type
>{};

}}}
#endif
