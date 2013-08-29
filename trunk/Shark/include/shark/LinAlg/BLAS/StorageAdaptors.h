//
//  Copyright (c) 2000-2006
//  Joerg Walter, Mathias Koch, Michael Stevens, Gunter Winkler
//
//  Permission to use, copy, modify, distribute and sell this software
//  and its documentation for any purpose is hereby granted without fee,
//  provided that the above copyright notice appear in all copies and
//  that both that copyright notice and this permission notice appear
//  in supporting documentation.  The authors make no representations
//  about the suitability of this software for any purpose.
//  It is provided "as is" without express or implied warranty.
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef BOOST_UBLAS_STORAGE_ADAPTORS_H
#define BOOST_UBLAS_STORAGE_ADAPTORS_H

#include <algorithm>

#include <shark/LinAlg/BLAS/Proxy.h>
#include <shark/LinAlg/BLAS/ublas.h>

namespace shark{ namespace blas{

	/** \brief converts a chunk of memory into a (readonly) usable ublas blas::vector.
	*
	* <code>
	*   double data[10]
	*   blas::vector<double> v(5);
	*   blas::matrix<double> m(5,10);
	*   v = prod(m, make_vector_from_pointer(10, &(data[0])));
	* </code>
	*/
	template <class T>
	FixedDenseVectorProxy<T> makeVector(const std::size_t size, T * data){
		return FixedDenseVectorProxy<T>(data,size);
	}
	
	template <class T, std::size_t N>
	FixedDenseVectorProxy<T> makeVector(T (&array)[N]){
		return FixedDenseVectorProxy<T>(array,N);
	}

	/** \brief converts a chunk of memory into a (readonly) usable dense blas::matrix.
	*
	* <code>
	*   double data[50]
	*   blas::vector<double> v(5);
	*   blas::vector<double> x(10);
	*   blas::matrix<double> m(5,10);
	*   v = prod(make_blas::matrix_from_pointer(5, 10, &(data[0])), x);
	* </code>
	*/
	template <class T>
	FixedDenseMatrixProxy<T> makeMatrix(const std::size_t size1, const std::size_t size2, T * data){
		return FixedDenseMatrixProxy<T>(data,size1, size2);
	}

	/// \brief converts a C-style 2D array into a (readonly) usable dense blas::matrix.
	template <class T, std::size_t M, std::size_t N>
	FixedDenseMatrixProxy<T> makeMatrix(T (&array)[M][N]){
		return FixedDenseMatrixProxy<T>(&(array[0][0]),M,N);
	}

}}

#endif
