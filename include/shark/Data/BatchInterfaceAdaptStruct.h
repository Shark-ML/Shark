/*!
 * 
 *
 * \brief       Defines an batch adptor for structures.
 * 
 * 
 *
 * \author      O.Krause
 * \date        2012
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
#ifndef SHARK_DATA_BATCHINTERFACEADAPTSTRUCT_H
#define SHARK_DATA_BATCHINTERFACEADAPTSTRUCT_H

#include <boost/fusion/sequence/intrinsic/swap.hpp>
#include <boost/fusion/algorithm/iteration/for_each.hpp>
#include <boost/fusion/algorithm/transformation/transform.hpp>
#include <shark/Data/BatchInterface.h>

#include <boost/preprocessor/seq/transform.hpp>
#include "Impl/BoostFusion151DefineStructInl.hpp"
namespace shark{
namespace detail{
///serializes the object into the archive
template<class Archive>
struct ItemSerializer {
	ItemSerializer(Archive& ar):m_ar(ar) {}

	template<typename T>
	void operator()(T& o)const{
		m_ar & o;
	}
private:
	Archive& m_ar;
};


struct CreateBatch{
	CreateBatch(std::size_t size):m_size(size) {}

	template<class> struct result;
	template<class T>
	struct result<CreateBatch(T const&)> {
		typedef typename shark::Batch<T>::type type;
	};

	template<class T>
	typename result<CreateBatch(T const&)>::type operator()(T const& value)const{
		return shark::Batch<T>::createBatch(value,m_size);
	}
private:
	std::size_t m_size;
};
struct resize{
	resize(std::size_t size1, std::size_t size2):m_size1(size1),m_size2(size2){};
	template<class T>
	void operator()(T& batch)const{
		 shark::Batch<typename boost::range_value<T>::type>::resize(batch,m_size1,m_size2);
	}
private:
	std::size_t m_size1;
	std::size_t m_size2;
};

///calls get(container,index) on a container. Used as boost fusion functor in the creation of references in the Batch Interface
struct MakeRef{
	template<class> struct result;
	template<class T>
	struct result<MakeRef(T const&)> {
		typedef typename boost::range_reference<T>::type type;
	};

	MakeRef(std::size_t index):m_index(index){}

	template<class T>
	typename result<MakeRef(T const&) >::type operator()(T const& container)const{
		return get(const_cast<T&>(container),m_index);//we need the const cast since the argument type must be a const ref.
	}
private:
	std::size_t m_index;
};
///calls get(container,index) on a container. Used as boost fusion functor in the cration of references in the Batch Interface
struct MakeConstRef{
	template<class> struct result;
	template<class T>
	struct result<MakeConstRef(T const&)> {
		typedef typename boost::range_reference<T const>::type type;
	};

	MakeConstRef(std::size_t index):m_index(index){}

	template<class T>
	typename result<MakeConstRef(T const&) >::type operator()(T const& container)const{
		return get(container,m_index);
	}
private:
	std::size_t m_index;
};

template<class FusionSequence>
struct FusionFacade: public FusionSequence{
	FusionFacade(){}
	template<class Sequence>
	FusionFacade(Sequence const& sequence):FusionSequence(sequence){}
};

template<class Type>
struct isFusionFacade{
private:
	struct Big{ int big[100]; };
	template <class S>
	static Big tester(FusionFacade<S>*);
	template <class S>
	static Big tester(FusionFacade<S> const*);
	static char tester(...);
	static Type* generator();

	BOOST_STATIC_CONSTANT(std::size_t, size = sizeof(tester(generator())));
public:
	BOOST_STATIC_CONSTANT(bool, value =  (size!= 1));
	typedef boost::mpl::bool_<value> type;
};

}

template<class S>
S& fusionize(detail::FusionFacade<S> & facade){
	return static_cast<S&>(facade);
}
template<class S>
S const& fusionize(detail::FusionFacade<S> const& facade){
	return static_cast<S const&>(facade);
}

template<class S>
typename boost::disable_if<detail::isFusionFacade<S>,S&>::type
fusionize(S& facade){
	return facade;
}
template<class S>
typename boost::disable_if<detail::isFusionFacade<S>,S const& >::type
fusionize(S const& facade){
	return facade;
}
}
#define SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL_IMPL(s,TYPE,ELEM)\
	( typename Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::TYPE,BOOST_PP_TUPLE_ELEM(2, 1, ELEM))

#define SHARK_TRANSFORM_TUPLELIST_IMPL(s, data,ELEM)\
	BOOST_PP_TUPLE_ELEM(2, 0, ELEM),BOOST_PP_TUPLE_ELEM(2, 1, ELEM)
#define SHARK_TRANSFORM_TUPLELIST(ELEMS)\
	BOOST_PP_SEQ_TRANSFORM(SHARK_TRANSFORM_TUPLELIST_IMPL, _ , ELEMS)

#define SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL(TYPE,ATTRIBUTES)\
	SHARK_TRANSFORM_TUPLELIST(BOOST_PP_SEQ_TRANSFORM(\
		SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL_IMPL,\
		TYPE, BOOST_PP_CAT(SHARK_FUSION_ADAPT_STRUCT_FILLER_0 ATTRIBUTES,_END)))

#define SHARK_TRANSFORM_BATCH_ATTRIBUTES_IMPL(s,TYPE,ELEM)\
	( Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::TYPE,BOOST_PP_TUPLE_ELEM(2, 1, ELEM))

#define SHARK_TRANSFORM_BATCH_ATTRIBUTES(TYPE,ATTRIBUTES)\
	SHARK_TRANSFORM_TUPLELIST(BOOST_PP_SEQ_TRANSFORM(\
		SHARK_TRANSFORM_BATCH_ATTRIBUTES_IMPL,\
		TYPE, BOOST_PP_CAT(SHARK_FUSION_ADAPT_STRUCT_FILLER_0 ATTRIBUTES,_END)))

///\brief creates default implementation for reference or const_reference types of Batches
#define SHARK_CREATE_BATCH_REFERENCES_TPL(ATTRIBUTES)\
private:\
SHARK_FUSION_DEFINE_STRUCT_REF_INLINE(FusionRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL(reference,ATTRIBUTES))\
SHARK_FUSION_DEFINE_STRUCT_CONST_REF_INLINE(FusionConstRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL(const_reference,ATTRIBUTES))\
public:\
struct reference: public detail::FusionFacade<FusionRef>{\
	template<class Batch>\
	reference(Batch& batch, std::size_t i)\
	:detail::FusionFacade<FusionRef>(boost::fusion::transform(fusionize(batch),detail::MakeRef(i))){}\
	template<class Other>\
	reference& operator= (Other const& other ){\
		fusionize(*this) = other;\
		return *this;\
	}\
	reference& operator= (reference const& other ){\
		fusionize(*this) = other;\
		return *this;\
	}\
	friend void swap(reference op1, reference op2){\
		boost::fusion::swap(op1,op2);\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(*this), ret);\
		return ret;\
	}\
};\
struct const_reference: public detail::FusionFacade<FusionConstRef>{\
private:\
const_reference& operator= (const_reference const& other );\
public:\
	template<class Batch>\
	const_reference(Batch& batch, std::size_t i)\
	:detail::FusionFacade<FusionConstRef>(boost::fusion::transform(fusionize(batch),detail::MakeConstRef(i))){}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(*this), ret);\
		return ret;\
	}\
};

#define SHARK_CREATE_BATCH_REFERENCES(ATTRIBUTES)\
private:\
SHARK_FUSION_DEFINE_STRUCT_REF_INLINE(FusionRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES(reference,ATTRIBUTES))\
SHARK_FUSION_DEFINE_STRUCT_CONST_REF_INLINE(FusionConstRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES(const_reference,ATTRIBUTES))\
public:\
struct reference: public detail::FusionFacade<FusionRef>{\
	template<class Batch>\
	reference(Batch& batch, std::size_t i)\
	:detail::FusionFacade<FusionRef>(boost::fusion::transform(fusionize(batch),detail::MakeRef(i))){}\
	template<class Other>\
	reference& operator= (Other const& other ){\
		fusionize(*this) = other;\
		return *this;\
	}\
	reference& operator= (reference const& other ){\
		fusionize(*this) = other;\
		return *this;\
	}\
	friend void swap(reference& op1, reference& op2){\
		boost::fusion::swap(op1,op2);\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(*this), ret);\
		return ret;\
	}\
};\
struct const_reference: public detail::FusionFacade<FusionConstRef>{\
	template<class Batch>\
	const_reference(Batch& batch, std::size_t i)\
	:detail::FusionFacade<FusionConstRef>(boost::fusion::transform(fusionize(batch),detail::MakeConstRef(i))){}\
	template<class Other>\
	const_reference& operator= (Other const& other ){\
		fusionize(*this) = other;\
		return *this;\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(*this), ret);\
		return ret;\
	}\
};

///\brief creates default typedefs for iterator or const_iterator types of Batches
#define SHARK_CREATE_BATCH_ITERATORS()\
typedef ProxyIterator<type, value_type, reference > iterator;\
typedef ProxyIterator<const type, value_type, const_reference > const_iterator;\
iterator begin(){\
	return iterator(*this,0);\
}\
const_iterator begin()const{\
	return const_iterator(*this,0);\
}\
iterator end(){\
	return iterator(*this,size());\
}\
const_iterator end()const{\
	return const_iterator(*this,size());\
}\
///\brief This macro can be used to specialize a structure type easily to a batch type.
///
///Assume, that your input Data looks like:
///<code>
///template<class T>
///struct DataType{
///     RealVector A;
///     T B;
///};
///</code>
///Than the Batch type should propably look like
///<code>
///struct DataTypeBatch{
///     RealMatrix A;
///     RealVector B;
///};
///</code>
///In this case the macro can be used to generate a complete specialisation of Batch<DataType>
///<code>
///#define DataVars (RealVector, A)(double B)
///
///SHARK_CREATE_BATCH_INTERFACE( DataType,DataVars)
///};
///As any other batch model th result also offers iterators over the range of elements.
///In this case also boost::fusion support is added to the sequence. e.g. it is
///handled similar to any other tuple type (RealMatrix,RealVector). This is useful for MKL or Transfer
///kernels
///</code>
#define SHARK_CREATE_BATCH_INTERFACE(NAME,ATTRIBUTES)\
private:\
	SHARK_FUSION_DEFINE_STRUCT_INLINE(FusionType, SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL(type,ATTRIBUTES))\
public:\
	struct type: public detail::FusionFacade<FusionType>{\
		typedef NAME value_type;\
		\
		SHARK_CREATE_BATCH_REFERENCES_TPL(ATTRIBUTES)\
		SHARK_CREATE_BATCH_ITERATORS()\
		\
		type(){}\
		type(std::size_t size1, std::size_t size2){\
			resize(size1,size2);\
		}\
		void resize(std::size_t batchSize, std::size_t elementSize){\
			boost::fusion::for_each(fusionize(*this), detail::resize(batchSize,elementSize));\
		}\
		\
		friend void swap(type& op1, type& op2){\
			boost::fusion::swap(fusionize(op1),fusionize(op2));\
		}\
		std::size_t size()const{\
			return shark::size(boost::fusion::at_c<0>(fusionize(*this)));\
		}\
		template<class Archive>\
		void serialize(Archive & archive,unsigned int version)\
		{\
			boost::fusion::for_each(fusionize(*this), detail::ItemSerializer<Archive>(archive));\
		}\
	};\
	typedef NAME value_type;\
	typedef typename type::reference reference;\
	typedef typename type::const_reference const_reference;\
	typedef typename type::iterator iterator;\
	typedef typename type::const_iterator const_iterator;\
	\
	static type createBatch(value_type const& input, std::size_t size = 1){\
		type batch;\
		boost::fusion::copy(boost::fusion::transform(input,detail::CreateBatch(size)),fusionize(batch));\
		return batch;\
	}\
	template<class Range>\
	static type createBatchFromRange(Range const& range){\
		std::size_t points = shark::size(range);\
		type batch = createBatch(*range.begin(),points);\
		typename boost::range_iterator<Range>::type pos = range.begin();\
		for(std::size_t i = 0; i != points; ++i,++pos){\
			get(batch,i) = *pos;\
		}\
		return batch;\
	}\
	static void resize(type& batch, std::size_t batchSize, std::size_t elements){\
		batch.resize(batchSize,elements);\
	}


///\brief This macro can be used to specialize a structure type easily to a batch type.
///
///Assume, thjat your input Data looks like:
///<code>
///struct DataType{
///     RealVector A;
///     double B;
///};
///</code>
///Than the Batch type should propably look like
///<code>
///struct DataTypeBatch{
///     RealMatrix A;
///     RealVector B;
///};
///</code>
///In this case the macro can be used to generate a complete specialisation of Batch<DataType>
///<code>
///#define DataVars (RealVector, A)(double B)
///
///SHARK_CREATE_BATCH_INTERFACE( DataType,DataVars)
///};
///As any other batch model the result also offers iterators over the range of elements.
///In this case also boost::fusion support is added to the sequence. e.g. it is
///handled similar to any other tuple type (RealMatrix,RealVector). This is useful for MKL or Transfer
///kernels
///</code>
#define SHARK_CREATE_BATCH_INTERFACE_NO_TPL(NAME,ATTRIBUTES)\
private:\
	SHARK_FUSION_DEFINE_STRUCT_INLINE(FusionType, SHARK_TRANSFORM_BATCH_ATTRIBUTES(type,ATTRIBUTES))\
public:\
	struct type: public detail::FusionFacade<FusionType>{\
		typedef NAME value_type;\
		\
		SHARK_CREATE_BATCH_REFERENCES(ATTRIBUTES)\
		SHARK_CREATE_BATCH_ITERATORS()\
		\
		type(){}\
		type(std::size_t size1, std::size_t size2){\
			resize(size1,size2);\
		}\
		void resize(std::size_t batchSize, std::size_t elementSize){\
			boost::fusion::for_each(fusionize(*this), detail::resize(batchSize,elementSize));\
		}\
		\
		friend void swap(type& op1, type& op2){\
			boost::fusion::swap(fusionize(op1),fusionize(op2));\
		}\
		std::size_t size()const{\
			return shark::size(boost::fusion::at_c<0>(fusionize(*this)));\
		}\
		template<class Archive>\
		void serialize(Archive & archive,unsigned int version)\
		{\
			boost::fusion::for_each(fusionize(*this), detail::ItemSerializer<Archive>(archive));\
		}\
	};\
	typedef NAME value_type;\
	typedef type::reference reference;\
	typedef type::const_reference const_reference;\
	typedef type::iterator iterator;\
	typedef type::const_iterator const_iterator;\
	\
	static type createBatch(value_type const& input, std::size_t size = 1){\
		type batch;\
		boost::fusion::copy(boost::fusion::transform(input,detail::CreateBatch(size)),fusionize(batch));\
		return batch;\
	}\
	template<class Range>\
	static type createBatchFromRange(Range const& range){\
		std::size_t points = shark::size(range);\
		type batch = createBatch(*range.begin(),points);\
		typename boost::range_iterator<Range>::type pos = range.begin();\
		for(std::size_t i = 0; i != points; ++i,++pos){\
			get(batch,i) = *pos;\
		}\
		return batch;\
	}\
	static void resize(type& batch, std::size_t batchSize, std::size_t elements){\
		batch.resize(batchSize,elements);\
	}

#endif
