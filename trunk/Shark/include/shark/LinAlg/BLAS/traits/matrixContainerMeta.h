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
//c_matrix
template<class T, std::size_t M, std::size_t N>
struct ExpressionTraitsBase<blas::c_matrix<T, M, N> >{
	typedef blas::c_matrix<T, M, N> type;
	typedef type const const_type;
	typedef T* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return M;
	}
	static std::size_t stride2(const_type&){
		return 1;
	}
};
template<class T, std::size_t M, std::size_t N>
struct ExpressionTraitsBase<blas::c_matrix<T, M, N> const>{
	typedef blas::c_matrix<T, M, N> const type;
	typedef type const const_type;
	typedef T const* value_pointer;
	
	typedef blas::row_major_tag orientation;
	typedef DenseStorage StorageCategory;
	static const bool transposed=false;
	
	static std::size_t stride1(const_type& m){
		return M;
	}
	static std::size_t stride2(const_type&){
		return 1;
	}
};

template<class T, std::size_t M, std::size_t N,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::c_matrix<T BOOST_PP_COMMA() M BOOST_PP_COMMA() N>)
	static value_pointer storageBegin(type& m){
		return &m.data().begin()[0];
	}
	static value_pointer storageEnd(type& m){
		return &m.data().begin()[0]+m.size1()*m.size2();
	}
};
//compressed_matrix
template<class T, std::size_t IB, class IA, class TA>
struct ExpressionTraitsBase<blas::compressed_matrix<T,blas::row_major,IB,IA,TA> >{
	
	typedef blas::compressed_matrix<T,blas::row_major,IB,IA,TA> type;
	typedef typename IA::value_type index_type;
	typedef typename TA::value_type value_type;
	typedef typename IA::value_type const* index_pointer;
	typedef typename TA::value_type const* value_pointer;
	typedef blas::row_major_tag orientation;
	
	static const std::size_t k = IB;
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
template<class T, std::size_t IB, class IA, class TA>
struct ExpressionTraitsBase<const blas::compressed_matrix<T,blas::row_major,IB,IA,TA> >
:public ExpressionTraitsBase<blas::compressed_matrix<T,blas::row_major,IB,IA,TA> >{};
	
//template<class T, std::size_t IB, class IA, class TA>
//struct ExpressionTraits<blas::compressed_matrix<T,blas::column_major,IB,IA,TA> >{
	
	//typedef blas::compressed_matrix<T,blas::column_major,IB,IA,TA> type;
	//typedef typename IA::value_type index_type;
	//typedef typename TA::value_type value_type;
	//typedef typename IA::value_type const* index_pointer;
	//typedef typename TA::value_type const* value_pointer;
	//typedef blas::column_major_tag orientation;
	
	//static const std::size_t k = IB;
	//static const StorageCategory category=Compressed;
	//static const bool transposed=false;
	
	//static std::size_t stride1(type const& m){
		//return m.size2();
	//}
	//static std::size_t stride2(type const&){
		//return 1;
	//}
	
	//static std::size_t index1Size(type const& v){
		//return v.filled1();
	//}
	//static std::size_t index2Size(type const& v){
		//return v.filled2();
	//}
	
	//static value_pointer storageBegin(type const& v){
		//return &v.value_data().begin()[0];
	//}
	//static value_pointer storageEnd(type const& v){
		//return &v.value_data().begin()[0];+v.nnz();
	//}
	//static index_pointer index1Begin(type const& v){
		//return &v.index1_data().begin()[0];
	//}
	//static index_pointer index1End(type const& v){
		//return &v.index1_data().begin()[0];+v.filled1();
	//}
	//static index_pointer index2Begin(type const& v){
		//return &v.index2_data().begin()[0];
	//}
	//static index_pointer index2End(type const& v){
		//return &v.index2_data().begin()[0];+v.filled2();
	//}
//};	
//template<class T, std::size_t IB, class IA, class TA>
//struct ExpressionTraits<const blas::compressed_matrix<T,blas::column_major,IB,IA,TA> >
//:public ExpressionTraits<blas::compressed_matrix<T,blas::column_major,IB,IA,TA> >{};
}}
#endif
