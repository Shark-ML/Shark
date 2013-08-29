/*!
 *  \author O. Krause
 *  \date 2012
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
#ifndef SHARK_LINALG_BLAS_TRAITS_MATRIXCONTAINERMETA_H
#define SHARK_LINALG_BLAS_TRAITS_MATRIXCONTAINERMETA_H

//this file is to included in metafunctions.h

namespace shark{
namespace blas{ 
namespace traits{
	
template<class T,class A>
struct ExpressionTraitsBase<blas::matrix<T, blas::row_major, A> >{
	typedef blas::matrix<T, blas::row_major, A> type;
	typedef type const const_type;
	typedef T* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return m.size2();
	}
	static std::size_t stride2(const_type&){
		return 1;
	}
};
template<class T,class A>
struct ExpressionTraitsBase<blas::matrix<T, blas::row_major, A> const>{
	typedef blas::matrix<T, blas::row_major, A> const type;
	typedef type const const_type;
	typedef T const* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return m.size2();
	}
	static std::size_t stride2(const_type&){
		return 1;
	}
};
template<class T,class A>
struct ExpressionTraitsBase<blas::matrix<T, blas::column_major, A> >{
	typedef blas::matrix<T, blas::column_major, A> type;
	typedef type const const_type;
	typedef T* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type&){
		return 1;
	}
	static std::size_t stride2(const_type& m){
		return m.size1();
	}
};
template<class T,class A>
struct ExpressionTraitsBase<blas::matrix<T, blas::column_major, A> const>{
	typedef blas::matrix<T, blas::column_major, A> const type;
	typedef type const const_type;
	typedef T const* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type&){
		return 1;
	}
	static std::size_t stride2(const_type& m){
		return m.size1();
	}
};

template<class T,class O,class A,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::matrix<T BOOST_PP_COMMA() O BOOST_PP_COMMA() A>)
	static value_pointer storageBegin(type& m){
		return &m.data().begin()[0];
	}
	static value_pointer storageEnd(type& m){
		return &m.data().begin()[0]+m.size1()*m.size2();
	}
};

//compressed_matrix
template<class T, class IA, class TA>
struct ExpressionTraitsBase<blas::compressed_matrix<T,blas::row_major,0,IA,TA> >{
	
	typedef blas::compressed_matrix<T,blas::row_major,0,IA,TA> type;
	typedef blas::compressed_matrix<T,blas::row_major,0,IA,TA> const const_type;
	typedef typename IA::value_type index_type;
	typedef typename TA::value_type value_type;
	typedef index_type const* index_pointer;
	typedef value_type const* value_pointer;
	typedef blas::row_major_tag orientation;
	
	typedef CompressedStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(type const&){
		return 1;
	}
	static std::size_t stride2(type const& m){
		return m.size1();
	}
	
	static std::size_t index1Size(type const& v){
		return v.filled1();
	}
	static std::size_t index2Size(type const& v){
		return v.filled2();
	}
};
template<class T, class IA, class TA>
struct ExpressionTraitsBase<blas::compressed_matrix<T,blas::row_major,0,IA,TA> const >{
	
	typedef blas::compressed_matrix<T,blas::row_major,0,IA,TA> const type;
	typedef type const_type;
	typedef typename IA::value_type index_type;
	typedef typename TA::value_type value_type;
	typedef index_type const* index_pointer;
	typedef value_type const* value_pointer;
	typedef blas::row_major_tag orientation;
	
	typedef CompressedStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(type const&){
		return 1;
	}
	static std::size_t stride2(type const& m){
		return m.size1();
	}
	
	static std::size_t index1Size(type const& v){
		return v.filled1();
	}
	static std::size_t index2Size(type const& v){
		return v.filled2();
	}
};
	
template<class T, class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(
	blas::compressed_matrix<T>
)
public:
	typedef CompressedMatrixStorage<T,std::size_t> storage;

	static storage compressedStorage(type& m){
		const_cast<blas::compressed_matrix<T>&>(m).complete_index1_data();
		storage matrixStorage;
		matrixStorage.data = &m.value_data()[0];
		matrixStorage.indizesLine = &m.index2_data()[0];
		matrixStorage.indizesLineBegin = &m.index1_data()[0];
		return matrixStorage;
	}
};

}}}
#endif
