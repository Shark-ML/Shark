/**
*
*  \brief Batch definitions for fusion::vectors which are used in MKL learning.
*
*  \author O.Krause, T.Glasmachers, T. Voss
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
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
#ifndef SHARK_DATA_MKLBATCHINTERFACE_H
#define SHARK_DATA_MKLBATCHINTERFACE_H

#include <shark/Data/BatchInterface.h>

#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/sequence/intrinsic/swap.hpp> 
#include <boost/fusion/algorithm/transformation/transform.hpp>
#include <boost/fusion/include/mpl.hpp>

#include <boost/mpl/transform.hpp>

namespace shark{

namespace detail{

template<class Reference,class Get>
class FusionVectorBatchReference:public Reference{
public:
	template<class Batch>
	FusionVectorBatchReference( Batch& batch,std::size_t i)
	:Reference(boost::fusion::transform(batch,Get(i))){}
	
//	#define SHARK_TRANSFORM_REF(z, n, _) get(boost::fusion::at_c<n>(batch),i) 
//	
//	template<class Batch>
//	FusionVectorBatchReference( Batch & batch, std::size_t i)
//	:Reference(BOOST_PP_ENUM(FUSION_MAX_VECTOR_SIZE, SHARK_TRANSFORM_REF, nil)){}
//	
//	#undef SHARK_TRANSFORM_REF
	
	template<class T> 
	FusionVectorBatchReference const& operator=(T const& vector){
		static_cast<Reference&>(*this)=vector;
		return *this;
	}
};

template<class Reference,class Get>
void swap(FusionVectorBatchReference<Reference,Get> ref1, FusionVectorBatchReference<Reference,Get> ref2){
	boost::fusion::swap(static_cast<Reference& >(ref1),static_cast<Reference& >(ref2));
}

}

/// \brief Default implementation for boost::fusion::vector.
///
/// if a sequence is of the Type vector< A, B, C > the BatchType is vector< Batch< A >,Batch< B >,Batch< C > >
/// We define this in it's own file, since boost::fusion is a real hard-hitter on compile-time.
template<BOOST_PP_ENUM_PARAMS(FUSION_MAX_VECTOR_SIZE, class T) >
struct Batch<boost::fusion::vector< BOOST_PP_ENUM_PARAMS(FUSION_MAX_VECTOR_SIZE, T)> >{
private:
	
	template<class U>
	struct BatchReference{
		typedef typename Batch<U>::reference type;
	};
	template<class U>
	struct BatchConstReference{
		typedef typename Batch<U>::const_reference type;
	};
	
	class Get
	{
	private:
		std::size_t m_index;
	public:
		Get(std::size_t index):m_index(index){}
		
		template<typename Sig>
		struct result;

		template<typename U>
		struct result<Get(U const&)>{
			typedef typename boost::iterator_reference<
				typename boost::range_iterator<U>::type
			>::type type;
		};

		template <typename U>
		typename result<Get(U const&)>::type operator()(U const& x) const
		{
			//fusion does not allow x to be non-const
			//return get(const_cast<U&>(x),m_index);
			return get(const_cast<U&>(x),m_index);
		}
	};
	
	class GetConst
	{
	private:
		std::size_t m_index;
	public:
		GetConst(std::size_t index):m_index(index){}
		
		template<typename Sig>
		struct result;

		template<typename U>
		struct result<GetConst(U const&)>{
			typedef typename boost::iterator_reference<
				typename boost::range_iterator<U const>::type
			>::type type;
		};

		template <typename U>
		typename result<GetConst(U const&)>::type operator()(U const& x) const
		{
			//fusion does not allow x to be non-const
			//return get(const_cast<U&>(x),m_index);
			return get(x,m_index);
		}
	};
	
	//boost::fusion functor which automatically creates a batch of a given size of a single element
	struct CreateBatch
	{
	private:
		std::size_t m_size;
	public:
		CreateBatch(std::size_t size):m_size(size){}
		
		template<typename Sig>
		struct result;

		template<typename U>
		struct result<CreateBatch(U)>: Batch<U>{};
		
		template <typename U>
		typename result<U>::type operator()(U const& x) const
		{
			return Batch<U>::createBatch(x,m_size);
		}
	};
public:

	/// \brief The type of the elements stored in the batch 
	typedef boost::fusion::vector< BOOST_PP_ENUM_PARAMS(FUSION_MAX_VECTOR_SIZE, T)> value_type;

	/// \brief Type of a batch of elements.
	typedef typename boost::mpl::transform<value_type,Batch<boost::mpl::_1> >::type type;

	//typedef typename boost::mpl::transform<T,BatchReference>::type reference;
	/// \brief Reference to a single element.
	typedef detail::FusionVectorBatchReference<typename boost::mpl::transform<value_type, BatchReference<boost::mpl::_1> >::type,Get> reference;
	/// \brief Type of a single immutable element.
	//typedef typename boost::mpl::transform<T,ConstBatchReference>::type const_reference;
	typedef detail::FusionVectorBatchReference<typename boost::mpl::transform<value_type, BatchConstReference<boost::mpl::_1> >::type,GetConst> const_reference;
	
	/// \brief The iterator type of the object.
	typedef ProxyIterator<type, value_type, reference > iterator;
	/// \brief The const_iterator type of the object.
	typedef ProxyIterator<type const, value_type, const_reference > const_iterator;
	
	
	///\brief creates a batch with several copies of the element
	static type createBatch(value_type const& input, std::size_t size = 1){
		return boost::fusion::transform(input,CreateBatch(size));
	}
	template<class Range>
	static type createBatch(Range const& range){
		//todo this is not optimal for sparse parts
		std::size_t points = shark::size(range);
		type batch = createBatch(*range.begin(),points);
		typename boost::range_iterator<Range>::type pos = range.begin();
		for(std::size_t i = 0; i != points; ++i){
			get(batch,i) = *pos;
			++pos;
		}
		return batch;
	}
};
}
#endif
