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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

///calls getBatchElement(container,index) on a container. Used as boost fusion functor in the creation of references in the Batch Interface
struct MakeRef{
	template<class> struct result;
	template<class T>
	struct result<MakeRef(T const&)> {
		typedef typename BatchTraits<T>::type::reference type;
	};

	MakeRef(std::size_t index):m_index(index){}

	template<class T>
	typename result<MakeRef(T const&) >::type operator()(T const& container)const{
		return getBatchElement(const_cast<T&>(container),m_index);//we need the const cast since the argument type must be a const ref.
	}
private:
	std::size_t m_index;
};
///calls getBatchElement(container,index) on a container. Used as boost fusion functor in the creation of references in the Batch Interface
struct MakeConstRef{
	template<class> struct result;
	template<class T>
	struct result<MakeConstRef(T const&)> {
		typedef typename BatchTraits<T>::type::const_reference type;
	};

	MakeConstRef(std::size_t index):m_index(index){}

	template<class T>
	typename result<MakeConstRef(T const&) >::type operator()(T const& container)const{
		return getBatchElement(container,m_index);
	}
private:
	std::size_t m_index;
};
}}

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
struct reference: public FusionRef{\
	template<class Batch>\
	reference(Batch& batch, std::size_t i)\
	:FusionRef(boost::fusion::transform(batch.fusionize(),detail::MakeRef(i))){}\
	template<class Other>\
	reference& operator= (Other const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	reference& operator= (reference const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	friend void swap(reference op1, reference op2){\
		boost::fusion::swap(op1,op2);\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(), ret);\
		return ret;\
	}\
	FusionRef& fusionize(){ return *this;}\
	FusionRef const& fusionize() const{ return *this;}\
};\
struct const_reference: public FusionConstRef{\
private:\
const_reference& operator= (const_reference const& other );\
public:\
	template<class Batch>\
	const_reference(Batch& batch, std::size_t i)\
	:FusionConstRef(boost::fusion::transform(batch.fusionize(),detail::MakeConstRef(i))){}\
	template<class Other>\
	const_reference& operator= (Other const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(), ret);\
		return ret;\
	}\
	FusionConstRef& fusionize(){ return *this;}\
	FusionConstRef const& fusionize() const{ return *this;}\
};

#define SHARK_CREATE_BATCH_REFERENCES(ATTRIBUTES)\
private:\
SHARK_FUSION_DEFINE_STRUCT_REF_INLINE(FusionRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES(reference,ATTRIBUTES))\
SHARK_FUSION_DEFINE_STRUCT_CONST_REF_INLINE(FusionConstRef, SHARK_TRANSFORM_BATCH_ATTRIBUTES(const_reference,ATTRIBUTES))\
public:\
struct reference: public FusionRef{\
	template<class Batch>\
	reference(Batch& batch, std::size_t i)\
	:FusionRef(boost::fusion::transform(batch.fusionize(),detail::MakeRef(i))){}\
	template<class Other>\
	reference& operator= (Other const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	reference& operator= (reference const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	friend void swap(reference op1, reference op2){\
		boost::fusion::swap(op1,op2);\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(), ret);\
		return ret;\
	}\
	FusionRef& fusionize(){ return *this;}\
	FusionRef const& fusionize() const{ return *this;}\
};\
struct const_reference: public FusionConstRef{\
	template<class Batch>\
	const_reference(Batch& batch, std::size_t i)\
	:FusionConstRef(boost::fusion::transform(batch.fusionize(),detail::MakeConstRef(i))){}\
	template<class Other>\
	const_reference& operator= (Other const& other ){\
		fusionize() = other;\
		return *this;\
	}\
	operator value_type()const{\
		value_type ret;\
		boost::fusion::copy(fusionize(), ret);\
		return ret;\
	}\
	FusionConstRef& fusionize(){ return *this;}\
	FusionConstRef const& fusionize() const{ return *this;}\
};

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
///In this case also boost::fusion support is added to the sequence. e.g. it is
///handled similar to any other tuple type (RealMatrix,RealVector). This is useful for MKL or Transfer
///kernels
///</code>
#define SHARK_CREATE_BATCH_INTERFACE(NAME,ATTRIBUTES)\
private:\
	SHARK_FUSION_DEFINE_STRUCT_INLINE(FusionType, SHARK_TRANSFORM_BATCH_ATTRIBUTES_TPL(type,ATTRIBUTES))\
public:\
	typedef std::array<Shape, boost::fusion::result_of::size< FusionType >::type::value > shape_type;\
	struct type: public FusionType{\
		typedef NAME value_type;\
		type(){}\
		\
		friend void swap(type& op1, type& op2){\
			boost::fusion::swap(op1.fusionize(),op2.fusionize());\
		}\
		std::size_t size()const{\
			return batchSize(boost::fusion::at_c<0>(fusionize()));\
		}\
		template<class Archive>\
		void serialize(Archive & archive,unsigned int version)\
		{\
			boost::fusion::for_each(fusionize(), detail::ItemSerializer<Archive>(archive));\
		}\
		FusionType& fusionize(){ return *this;}\
		FusionType const& fusionize() const{ return *this;}\
	};\
	typedef NAME value_type;\
	SHARK_CREATE_BATCH_REFERENCES_TPL(ATTRIBUTES)\
	\
	static type createBatch(value_type const& input, std::size_t size = 1){\
		type batch;\
		boost::fusion::copy(boost::fusion::transform(input,detail::CreateBatch(size)),batch.fusionize());\
		return batch;\
	}\
	template<class Iterator>\
	static type createBatchFromRange(Iterator const& begin, Iterator const& end){\
		std::size_t points = end - begin;\
		type batch = createBatch(*begin,points);\
		Iterator pos = begin;\
		for(std::size_t i = 0; i != points; ++i,++pos){\
			getBatchElement(batch,i) = *pos;\
		}\
		return batch;\
	}\
	static type createBatchFromShape(shape_type const& shape, std::size_t size = 1){\
		throw SHARKEXCEPTION("NOT IMPLEMENTED");\
	}\
	template<class T>\
	static std::size_t size(T const& batch){return batch.size();}\
	\
	static reference get(type& batch, std::size_t i){\
		return reference(batch,i);\
	}\
	static const_reference get(type const& batch, std::size_t i){\
		return const_reference(batch,i);\
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
///In this case also boost::fusion support is added to the sequence. e.g. it is
///handled similar to any other tuple type (RealMatrix,RealVector). This is useful for MKL or Transfer
///kernels
///</code>
#define SHARK_CREATE_BATCH_INTERFACE_NO_TPL(NAME,ATTRIBUTES)\
private:\
	SHARK_FUSION_DEFINE_STRUCT_INLINE(FusionType, SHARK_TRANSFORM_BATCH_ATTRIBUTES(type,ATTRIBUTES))\
public:\
	typedef std::array<Shape, boost::fusion::result_of::size< FusionType >::type::value > shape_type;\
	struct type: public FusionType{\
		typedef NAME value_type;\
		type(){}\
		\
		friend void swap(type& op1, type& op2){\
			boost::fusion::swap(op1.fusionize(),op2.fusionize());\
		}\
		std::size_t size()const{\
			return batchSize(boost::fusion::at_c<0>(fusionize()));\
		}\
		template<class Archive>\
		void serialize(Archive & archive,unsigned int version)\
		{\
			boost::fusion::for_each(fusionize(), detail::ItemSerializer<Archive>(archive));\
		}\
		FusionType& fusionize(){ return *this;}\
		FusionType const& fusionize() const{ return *this;}\
	};\
	typedef NAME value_type;\
	SHARK_CREATE_BATCH_REFERENCES(ATTRIBUTES)\
	\
	static type createBatch(value_type const& input, std::size_t size = 1){\
		type batch;\
		boost::fusion::copy(boost::fusion::transform(input,detail::CreateBatch(size)),batch.fusionize());\
		return batch;\
	}\
	template<class Iterator>\
	static type createBatchFromRange(Iterator const& begin, Iterator const& end){\
		std::size_t points = end - begin;\
		type batch = createBatch(*begin,points);\
		Iterator pos = begin;\
		for(std::size_t i = 0; i != points; ++i,++pos){\
			getBatchElement(batch,i) = *pos;\
		}\
		return batch;\
	}\
	static std::size_t size(type const& batch){return batch.size();}\
	\
	static reference get(type& batch, std::size_t i){\
		return reference(batch,i);\
	}\
	static const_reference get(type const& batch, std::size_t i){\
		return const_reference(batch,i);\
	}

#endif
