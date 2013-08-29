//===========================================================================
/*!
 *  \author O. Krause
 *  \date 2013
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
//  Based on the boost::numeric bindings
//  Copyright (c) 2002,2003,2004
//  Toon Knapen, Kresimir Fresl, Joerg Walter, Karl Meerbergen
//
// Distributed under the Boost Software License, Version 1.0. 
// (See accompanying file LICENSE_1_0.txt or copy at 
// http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef SHARK_LINALG_BLAS_TRAITS_METAFUNCTIONS_H
#define SHARK_LINALG_BLAS_TRAITS_METAFUNCTIONS_H

#include <shark/Core/Exception.h>
#include <shark/Core/utility/CopyConst.h>
#include <shark/LinAlg/BLAS/ublas.h>

#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/inherit.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
namespace shark { namespace blas{ namespace traits {
	
struct UnknownStorage{};
struct CompressedStorage{};
struct DenseStorage{};

template<class ValueType, class IndexType>
struct CompressedVectorStorage{
	typedef ValueType value_type;
	typedef IndexType index_type;
	ValueType const* data;
	IndexType const* indizes;
	std::size_t nonZeros;
	std::size_t startIndex;
};

template<class ValueType, class IndexType>
struct CompressedMatrixStorage{
	typedef ValueType value_type;
	typedef IndexType index_type;
	ValueType const* data;
	IndexType const* indizesLine;
	IndexType const* indizesLineBegin;
};
	
//explanations:
//stride: distance in memory between two successive positions of the dimension
//storage: beginning of the memory of the expression
//category: the type of underlying storage

template<class T>
struct ExpressionTraitsBase{
	typedef UnknownStorage StorageCategory;
};
template<class T,class BaseExpression>
struct DenseTraitsImpl{};

template<class T,class BaseExpression>
struct CompressedTraitsImpl{};
	
#define SHARK_DENSETRAITSSPEC(Expression)\
struct DenseTraitsImpl<Expression,BaseExpression>\
:public ExpressionTraitsBase<BaseExpression>{\
private:\
	typedef  ExpressionTraitsBase<BaseExpression> base_type;\
public:\
	typedef typename base_type::type type;\
	typedef typename base_type::const_type const_type;\
	typedef typename base_type::value_pointer value_pointer;

#define SHARK_COMPRESSEDTRAITSSPEC(Expression)\
struct CompressedTraitsImpl<Expression,BaseExpression>\
:public ExpressionTraitsBase<BaseExpression>{\
private:\
	typedef  ExpressionTraitsBase<BaseExpression> base_type;\
public:\
	typedef typename base_type::type type;\
	typedef typename base_type::const_type const_type;\
	typedef typename base_type::value_pointer value_pointer;

template<class T>
struct DenseTraits
:public DenseTraitsImpl<typename boost::remove_const<T>::type,T>{};

template<class T>
struct CompressedTraits
:public CompressedTraitsImpl<typename boost::remove_const<T>::type,T>{};

//////////////VECTOR PROXY EXPRESSIONS///////////////////////////

//vector expression
template<class V>
struct ExpressionTraitsBase<blas::vector_expression<V> >{
private:
	typedef ExpressionTraitsBase<V> Traits;
public:
	typedef blas::vector_expression<V > type;
	typedef type const const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v());
	}
};
template<class V>
struct ExpressionTraitsBase<blas::vector_expression<V> const >{
private:
	typedef ExpressionTraitsBase<V const> Traits;
public:
	typedef blas::vector_expression<V > const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v());
	}
};

template<class V,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::vector_expression<V>)
private:
	typedef DenseTraits<V> Traits;
public:
	static value_pointer storageBegin(type& v){
		return Traits::storageBegin(v());
	}
	static value_pointer storageEnd(type& v){
		return Traits::storageEnd(v());
	}
};
template<class V,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::vector_expression<V>)
	typedef CompressedTraits<typename CopyConst<V,BaseExpression>::type> Traits;
public:
	typedef typename Traits::storage storage;

	static storage compressedStorage(type& v){
		return Traits::compressedStorage(v);
	}

};

//vector reference
template<class V>
struct ExpressionTraitsBase<blas::vector_reference<V> >{
private:
	typedef ExpressionTraitsBase<V> Traits;
public:
	typedef blas::vector_reference<V> type;
	typedef type const const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v());
	}
};
template<class V>
struct ExpressionTraitsBase<blas::vector_reference<V> const >{
private:
	typedef ExpressionTraitsBase<V const> Traits;
public:
	typedef blas::vector_reference<V > const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v.expression());
	}
};

template<class V,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::vector_reference<V>)
private:
	typedef DenseTraits<typename CopyConst<V,BaseExpression>::type > Traits;
public:
	static value_pointer storageBegin(type& v){
		return Traits::storageBegin(v.expression());
	}
	static value_pointer storageEnd(type& v){
		return Traits::storageEnd(v.expression());
	}
};
template<class V,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::vector_reference<V>)
private:
	typedef CompressedTraits<typename CopyConst<V,BaseExpression>::type> Traits;
public:
	typedef typename Traits::storage storage;

	static storage compressedStorage(type& v){
		return Traits::compressedStorage(v);
	}
};

//vector range
template<class V>
struct ExpressionTraitsBase<blas::vector_range<V> >{
private:
	typedef ExpressionTraitsBase<typename blas::vector_range<V>::vector_closure_type const> Traits;
public:
	typedef blas::vector_range<V> type;
	typedef blas::vector_range<V> const const_type;
	typedef typename ExpressionTraitsBase<V>::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v.data());
	}
};
template<class V>
struct ExpressionTraitsBase<blas::vector_range<V> const>{
private:
	typedef ExpressionTraitsBase<typename blas::vector_range<V>::vector_closure_type const> Traits;
public:
	typedef blas::vector_range<V> const type;
	typedef type const_type;
	typedef typename ExpressionTraitsBase<V>::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride(v.data());
	}
};

template<class V,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::vector_range<V>)
private:
	typedef typename blas::vector_range<V>::vector_closure_type Proxy;
	typedef DenseTraits<Proxy> Traits;
public:
	static value_pointer storageBegin(blas::vector_range<V> v){
		return Traits::storageBegin(v.data())+v.start() * base_type::stride(v);
	}
	static value_pointer storageEnd(blas::vector_range<V> v){
		return storageBegin(v)+v.size() * base_type::stride(v);
	}
};
template<class V,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::vector_range<V>)
private:
	typedef typename type::vector_closure_type Proxy;
	typedef typename CopyConst<Proxy,BaseExpression>::type CProxy;
	typedef CompressedTraits<CProxy> Traits;
public:
	typedef typename Traits::storage storage;

	static storage compressedStorage(type& v){
		SHARK_CHECK(Traits::stride(v) != 1, 
		"[Compressed Vector Traits]:can't adapt compressed storage for vectors with stride != 1");
		
		storage rangeStorage = Traits::compressedStorage(v.data());
		rangeStorage.startIndex += v.start();
		std::size_t endIndex = rangeStorage.startIndex+v.size();
		
		//find the first index of the array
		while(rangeStorage.nonZeros && *rangeStorage.indizes < rangeStorage.startIndex){
			--rangeStorage.nonZeros;
			++rangeStorage.indizes;
			++rangeStorage.values;
		}
		//check, whether the resulting range is empty
		if(!rangeStorage.nonZeros
		|| *rangeStorage.indizes >= endIndex //if the first element is already bigger than the last
		){
			rangeStorage.nonZeros = 0;
			return rangeStorage;
		}
		//find end to get the correct number of nonzeros
		while( rangeStorage.indizes[rangeStorage.nonZeros-1] >= endIndex)
			--rangeStorage.nonZeros;
		return rangeStorage;
	}
};

//////////////MATRIX PROXY EXPRESSIONS///////////////////////

//matrix row
template<class M>
struct ExpressionTraitsBase<blas::matrix_row<M> >{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_row<M>::matrix_closure_type const> Traits;
public:
	typedef blas::matrix_row<M> type;
	typedef type const const_type;
	typedef typename ExpressionTraitsBase<M>::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride2(v.data());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_row<M> const>{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_row<M>::matrix_closure_type const> Traits;
public:
	typedef blas::matrix_row<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride2(v.data());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_row<M>)
private:
	typedef typename type::matrix_closure_type Proxy;
	typedef typename CopyConst<Proxy,BaseExpression>::type CProxy;
	typedef DenseTraits<CProxy> Traits;
public:
	static value_pointer storageBegin(type& v){
		CProxy proxy = v.data();
		return Traits::storageBegin(proxy)+v.index() * Traits::stride1(proxy);
	}
	static value_pointer storageEnd(type& v){
		CProxy proxy = v.data();
		return storageBegin(v)+v.size() * Traits::stride1(proxy);
	}
};
template<class M,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::matrix_row<M>)
private:
	typedef typename type::matrix_closure_type Proxy;
	typedef typename CopyConst<Proxy,BaseExpression>::type CProxy;
	typedef CompressedTraits<CProxy> Traits;
	static const bool MatIsRowMajor=boost::is_same<typename Traits::orientation,blas::row_major_tag>::value;
	typedef typename Traits::storage matrix_storage;
public:
	typedef CompressedVectorStorage<
		typename matrix_storage::value_type,
		typename matrix_storage::index_type
	> storage;

	static storage compressedStorage(type& v){
		SHARK_CHECK(MatIsRowMajor, 
		"[Compressed Vector Traits]:It is not possible to adapt compressed storage of rows from column major matrices");
		
		//get matriy storage
		CProxy proxy = v.data();
		matrix_storage matrixStorage = Traits::compressedStorage(proxy);
		
		//get the row
		std::size_t startIndex = matrixStorage.indizesLineBegin[v.index()];
		std::size_t endIndex = matrixStorage.indizesLineBegin[v.index()+1];
		storage vectorStorage;
		vectorStorage.indizes = matrixStorage.indizesLine+startIndex;
		vectorStorage.data = matrixStorage.data+startIndex;
		vectorStorage.nonZeros = endIndex-startIndex;
		vectorStorage.startIndex= 0;//we ar at the beginning of the line
		return vectorStorage;
	}
};

//matrix column
template<class M>
struct ExpressionTraitsBase<blas::matrix_column<M> >{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_column<M>::matrix_closure_type const> Traits;
public:
	typedef blas::matrix_column<M> type;
	typedef type const const_type;
	typedef typename ExpressionTraitsBase<M>::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride1(v.data());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_column<M> const>{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_column<M>::matrix_closure_type const> Traits;
public:
	typedef blas::matrix_column<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	
	typedef typename Traits::StorageCategory StorageCategory;
	
	static std::size_t stride(const_type& v){
		return Traits::stride1(v.data());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_column<M>)
private:
	typedef typename type::matrix_closure_type Proxy;
	typedef typename CopyConst<Proxy,BaseExpression>::type CProxy;
	typedef DenseTraits<CProxy> Traits;
public:
	static value_pointer storageBegin(type& v){
		return Traits::storageBegin(v.data())+v.index() * Traits::stride2(v.data());
	}
	static value_pointer storageEnd(type& v){
		return storageBegin(v)+v.size() * Traits::stride2(v.data());
	}
};

//matrix expression
template<class M>
struct ExpressionTraitsBase<blas::matrix_expression<M> >{
private:
	typedef ExpressionTraitsBase<M> Traits;
public:
	typedef blas::matrix_expression<M> type;
	typedef type const const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_expression<M> const>{
private:
	typedef ExpressionTraitsBase<M> Traits;
public:
	typedef blas::matrix_expression<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_expression<M>)
private:
	typedef DenseTraits<typename CopyConst<M,BaseExpression>::type> Traits;
public:
	static value_pointer storageBegin(type& m){
		return Traits::storageBegin(m());
	}
	static value_pointer storageEnd(type & m){
		return Traits::storageEnd(m());
	}
};

template<class M,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::matrix_expression<M>)
private:
	typedef DenseTraits<typename CopyConst<M,BaseExpression>::type> Traits;
public:
	typedef typename Traits::storage storage;

	static storage compressedStorage(type& m){
		return Traits::compressedStorage(m());
	}
};
//matrix reference
template<class M>
struct ExpressionTraitsBase<blas::matrix_reference<M> >{
private:
	typedef ExpressionTraitsBase<M> Traits;
public:
	typedef blas::matrix_reference<M> type;
	typedef type const const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m.expression());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m.expression());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_reference<M> const>{
private:
	typedef ExpressionTraitsBase<M const> Traits;
public:
	typedef blas::matrix_reference<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m.expression());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m.expression());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_reference<M>)
private:
	typedef typename CopyConst<M,BaseExpression>::type MatrixType;
	typedef DenseTraits<MatrixType> Traits;
public:
	static value_pointer storageBegin(type& m){
		return Traits::storageBegin(const_cast<MatrixType&>(m.expression()));
	}
	static value_pointer storageEnd(type & m){
		return Traits::storageEnd(const_cast<MatrixType&>(m.expression()));
	}
};

template<class M,class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(blas::matrix_reference<M>)
private:
	typedef typename CopyConst<M,BaseExpression>::type MatrixType;
	typedef CompressedTraits<MatrixType> Traits;
public:
	typedef typename Traits::storage storage;

	static storage compressedStorage(type& m){
		return Traits::compressedStorage(const_cast<MatrixType&>(m.expression()));
	}
};

//matrix transpose
template<class M>
struct ExpressionTraitsBase<blas::matrix_transpose<M> >{
private:
	typedef ExpressionTraitsBase<
		typename blas::matrix_transpose<M>::expression_closure_type const
	> Traits;
public:
	typedef blas::matrix_transpose<M> type;
	typedef type const const_type;
	typedef typename ExpressionTraitsBase<M>::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=!Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride2(m.expression());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride1(m.expression());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_transpose<M> const>{
private:
	typedef ExpressionTraitsBase<
		typename blas::matrix_transpose<M>::expression_closure_type const
	> Traits;
public:
	typedef blas::matrix_transpose<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=!Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride2(m.expression());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride1(m.expression());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_transpose<M>)
private:
	typedef DenseTraits<
		typename blas::matrix_transpose<M>::expression_closure_type const
	> Traits;
public:
	static value_pointer storageBegin(type& m){
		return const_cast<value_pointer>(Traits::storageBegin(m.expression()));
	}
	static value_pointer storageEnd(type& m){
		return const_cast<value_pointer>(Traits::storageEnd(m.expression()));
	}
};

//matrix range
template<class M>
struct ExpressionTraitsBase<blas::matrix_range<M> >{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_range<M>::matrix_closure_type> Traits;
public:
	typedef blas::matrix_range<M> type;
	typedef type const const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m.data());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m.data());
	}
};
template<class M>
struct ExpressionTraitsBase<blas::matrix_range<M> const>{
private:
	typedef ExpressionTraitsBase<typename blas::matrix_range<M>::matrix_closure_type const> Traits;
public:
	typedef blas::matrix_range<M> const type;
	typedef type const_type;
	typedef typename Traits::value_pointer value_pointer;
	typedef typename Traits::orientation orientation;
	
	typedef typename Traits::StorageCategory StorageCategory;
	static const bool transposed=Traits::transposed;
	
	static std::size_t stride1(const_type& m){
		return Traits::stride1(m.data());
	}
	static std::size_t stride2(const_type& m){
		return Traits::stride2(m.data());
	}
};

template<class M,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix_range<M>)
private:
	typedef typename blas::matrix_range<M>::matrix_closure_type Closure;
	typedef DenseTraits<typename CopyConst<Closure,BaseExpression>::type > Traits;
public:
	static value_pointer storageBegin(type& m){
		return Traits::storageBegin(m.data())
		+ m.start1() * Traits::stride1(m.data())
		+ m.start2() * Traits::stride2(m.data());
	}
	static value_pointer storageEnd(type& m){
		return storageBegin(m.data())
		+ m.size1() * Traits::stride1(m.data())
		+ m.size2() * Traits::stride2(m.data());
	}
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////METAFUNCTIONS BUILD ON TOP OF EXPRESSIONTRAITSBASE//////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<class E>
struct IsDense
: public boost::is_same<typename ExpressionTraitsBase<E>::StorageCategory,DenseStorage>{};

template<class E>
struct IsUnknownStorage
: public boost::is_same<typename ExpressionTraitsBase<E>::StorageCategory,UnknownStorage>{};

template<class E>
struct IsCompressed
: public boost::is_same<typename ExpressionTraitsBase<E>::StorageCategory,CompressedStorage>{};

template<class E>
struct IsSparse{
	static const bool value = !IsDense<E>::value;
	typedef boost::mpl::bool_<value> type;
};

template<class Expression>
class ExpressionTraits:
public boost::mpl::inherit<
	typename boost::mpl::if_<IsDense<Expression>,DenseTraits<Expression>,boost::mpl::empty_base>::type,
	typename boost::mpl::if_<IsCompressed<Expression>,CompressedTraits<Expression>,boost::mpl::empty_base>::type
>::type
{};
	
template<class E> 
struct PointerType{
	//~ typedef typename ExpressionTraits<E>::value_pointer type;
	typedef typename DenseTraitsImpl<typename boost::remove_const<E>::type,E>::value_pointer type;
};
template<class M> 
struct Orientation{
	typedef typename ExpressionTraits<M>::orientation type;
};



}}}

#include "vectorContainerMeta.h"
#include "matrixContainerMeta.h"
#endif
