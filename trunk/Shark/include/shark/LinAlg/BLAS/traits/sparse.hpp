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

#ifndef SHARK_LINALG_BLAS_TRAITS_SPARSE_HPP
#define SHARK_LINALG_BLAS_TRAITS_SPARSE_HPP

#include "vector_raw.hpp"
#include "matrix_raw.hpp"


namespace shark { namespace blas{  namespace traits {

//O.K. needs rework!
//template<class ValueType, class IndexType>
//struct CompressedVectorStorage{
//	ValueType const* data;
//	IndexType const* indizes;
//	std::size_t nonZeros;
//};
//template<class ValueType, class IndexType>
//struct CompressedMatrixStorage{
//	ValueType const* data;
//	IndexType const* indizes1;
//	IndexType const* indizes2;
//	std::size_t nonZeros;
//	std::size_t leadingIndizes;
//}
//template<class V>
//struct VecStorageType{
//	typedef CompressedTraits<M> traits;
//	typedef typename traits::index_type index_type;
//	typedef typename traits::value_type value_type;
//	typedef CompressedVectorStorage<value_type, index_type> type;
//};
//template<class V>
//struct MatStorageType{
//	typedef CompressedTraits<M> traits;
//	typedef typename traits::index_type index_type;
//	typedef typename traits::value_type value_type;
//	typedef CompressedMatrixStorage<value_type, index_type> type;
//};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////COMPRESSED DIRECT STORAGE ACCESSORS////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////forward declaration for the general versions of the methods so that the later methods will find them
//template<class M>
//typename MatStorageType<M>::type compressedData(blas::matrix_expression<M> const& m);
//	
/////////////////////////COMPRESSED VECTOR ACCESSORS///////////////////////////////

////returns the continuous array of data to the nonzero-arrays
//template<class T, std::size_t IB, class IA, class TA>
//typename VecStorageType<blas::compressed_vector<T,IB,IA,TA> >::type 
//compressedVectorStorageImpl(blas::compressed_vector<T,IB,IA,TA> const& v){
//	typename VecStorageType<blas::compressed_vector<T,IB,IA,TA> >::type  storage={
//		&(v.value_data()[0])+IB,
//		&(v.index_data()[0])+IB,
//		v.nnz()
//	};
//	return storage;
//}
////the following are the usual peel offs for all supported expressions
////in the compressed case, it is a bit harder, since we have to find the actual positions in the arrays
////also there are combinations which are simply not suitable for direct memory access. e.g. taking columns from a rom_major matrix
////in this case, we return an empty position. no compile errors since this can be handled more easily in the normal code stream

//template<V>
//typename VecStorageType<V>::type compressedDataImpl(blas::vector_range<V> const& v){
//	typename VecStorageType<V>::type storage=compressedData(v.data());
//	std::size_t start=v.start();
//	while(*storage.indizes < start && storage.nonZeros){
//		++storage.indizes;
//		++storage.data;
//		--storage.nonzeros;
//	}
//	return storage;
//}

////reference
//template<class V>
//typename VecStorageType<V>::type compressedDataImpl(blas::vector_reference<V> const& v){
//	return compressedData(v.expression());
//}

////matrix row. we have to take into account, that we can only take rows from row_major matrices
//template<class M>
//typename VecStorageType<M>::type compressedDataImpl(blas::matrix_row<M> const& m){
//	typename MatStorageType<V>::type matStorage = compressedData(m.data());
//	SHARK_CHECK(isRowMajor(m.data()) && matStorage.leadingIndizes,"LOGIC_ERROR! compressedStorage for matrix_rows is only allowed when the matrix is row_major");
//	
//	typename VecStorageType<V>::type rowStorage;
//		
//	//find correct starting index
//	std::size_t pos = 0;
//	while(pos != matStorage.leadingIndizes && matStorage.indizes2[pos]<m.index()){
//		++pos;
//	}
//	//get the beginning to the row at index pos
//	rowStorage.indizes = matStorage.indizes1+matStorage.indizes2[pos];
//	rowStorage.data = matStorage.data+matStorage.indizes2[pos];
//	if(pos != leadingIndizes)
//		rowStorage.nonZeros=matStorage.indizes2[pos+1]-matStorage.indizes2[pos];
//	else
//		rowStorage.nonZeros=matStorage.nnz-matStorage.indizes2[pos];
//	return rowStorage;
//}

////matrix column. we have to take into account, that we can only take rows from row_major matrices
//template<class M>
//typename VecStorageType<M>::type compressedDataImpl(blas::matrix_column<M> const& m){
//	typename MatStorageType<V>::type matStorage = compressedData(m.data());
//	SHARK_CHECK(isColumnMajor(m.data()) && matStorage.leadingIndizes,"LOGIC_ERROR! compressedStorage for matrix_columns is only allowed when the matrix is column_major");
//	
//	typename VecStorageType<V>::type rowStorage;
//		
//	//find correct starting index
//	std::size_t pos = 0;
//	while(pos != matStorage.leadingIndizes && matStorage.indizes2[pos]<m.index()){
//		++pos;
//	}
//	//get the beginning to the row at index pos
//	rowStorage.indizes = matStorage.indizes1+matStorage.indizes2[pos];
//	rowStorage.data = matStorage.data+matStorage.indizes2[pos];
//	if(pos != leadingIndizes)
//		rowStorage.nonZeros=matStorage.indizes2[pos+1]-matStorage.indizes2[pos];
//	else
//		rowStorage.nonZeros=matStorage.nnz-matStorage.indizes2[pos];
//	return rowStorage;
//}

////everything stranding here is unfortunately not allowed
//template<class V>
//typename VecStorageType<V>::type compressedDataImpl(V const& m){
//	SHARK_CHECK(false,"LOGIC_ERROR! unsupported argument expression");
//	//prevent warnings from MR. compiler.
//	typename VecStorageType<V>::type storage;
//	return storage;
//}

////general version.. returns the data of a valid compressed vector expression. Only produces runtime errors
//template<class V>
//typename VecStorageType<V>::type compressedData(blas::vector_expression<V> const& v){
//	return compressedDataImpl(v());
//}

/////////////////////////COMPRESSED MATRIX STORAGE ACCESSORS///////////////////////////////
//template<class T, L, std::size_t IB, class IA, class TA>
//typename MatStorageType<blas::compressed_matrix<T,L, IB,IA,TA> >::type 
//compressedData(blas::compressed_matrix<T,L, IB,IA,TA> const& v){
//	typename MatStorageType<M>::type storage={
//		&(v.value_data()[0])+IB,
//		&(v.index_1data()[0])+IB,
//		&(v.index_2data()[0])+IB,
//		v.nnz(),
//		v.filled1()
//	};
//	return storage;
//}
////matrix transposition and similar constructs
//template<class M, class F>
//class IsCompressed<blas::matrix_unary2<M, F > >
//:public IsCompressed<M>{};

}}}

#endif
