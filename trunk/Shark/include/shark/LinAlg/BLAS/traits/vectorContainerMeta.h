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
#ifndef SHARK_LINALG_TRAITS_VECTORCONTAINERMETA_H
#define SHARK_LINALG_TRAITS_VECTORCONTAINERMETA_H

//this file is to included in metafunctions.h

namespace shark{
namespace blas{ 
namespace traits{

//vector
template<class T,class A>
struct ExpressionTraitsBase<blas::vector<T,A > >{
	typedef blas::vector<T,A > type;
	typedef type const const_type;
	typedef T* value_pointer;
	
	typedef DenseStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return 1;
	}
};
template<class T,class A>
struct ExpressionTraitsBase<blas::vector<T,A > const>{
	typedef blas::vector<T,A > const type;
	typedef type const const_type;
	typedef T const* value_pointer;
	
	typedef DenseStorage StorageCategory;
	
	static std::size_t stride(const_type& v){
		return 1;
	}
};

template<class T,class A,class BaseExpression>
SHARK_DENSETRAITSSPEC(blas::vector<T BOOST_PP_COMMA() A >)
	static value_pointer storageBegin(type& v){
		return &v.data().begin()[0];
	}
	static value_pointer storageEnd(type& v){
		return &v.data().begin()[0];
	}
};
//compressed_vector
template<class T, std::size_t IB, class IA, class TA>
struct ExpressionTraitsBase<blas::compressed_vector<T,IB, IA, TA > >{
	
	typedef blas::compressed_vector<T,IB, IA, TA > type;
	typedef type const const_type;
	typedef typename IA::value_type index_type;
	typedef typename TA::value_type value_type;
	typedef typename IA::value_type const* index_pointer;
	typedef typename TA::value_type const* value_pointer;
	
	static const std::size_t k = IB;
	typedef CompressedStorage StorageCategory;
	
	static std::size_t stride(type const&){
		return 1;
	}
};
template<class T, std::size_t IB, class IA, class TA>
struct ExpressionTraitsBase<blas::compressed_vector<T,IB, IA, TA > const>{
	typedef blas::compressed_vector<T,IB, IA, TA > const type;
	typedef type const_type;
	typedef typename IA::value_type index_type;
	typedef typename TA::value_type value_type;
	typedef typename IA::value_type const* index_pointer;
	typedef typename TA::value_type const* value_pointer;
	
	static const std::size_t k = IB;
	typedef CompressedStorage StorageCategory;
	
	static std::size_t stride(type const&){
		return 1;
	}
};

template<class T, class BaseExpression>
SHARK_COMPRESSEDTRAITSSPEC(
	blas::compressed_vector<T>
)
public:
	typedef CompressedVectorStorage<T,std::size_t> storage;

	static storage compressedStorage(type& v){
		storage vectorStorage;
		vectorStorage.data = &v.value_data()[0];
		vectorStorage.indizes = &v.index_data()[0];
		vectorStorage.nonZeros = v.nnz();
		vectorStorage.startIndex = 0;
		return vectorStorage;
	}
};

}}}
#endif
