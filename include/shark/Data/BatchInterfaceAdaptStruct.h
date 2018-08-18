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

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/comma_if.hpp>
#include <boost/preprocessor/seq/transform.hpp>

#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/seq/seq.hpp>
#include <boost/preprocessor/tuple/elem.hpp>


//transforms a tuple list (A, a) (B, b)... into a list of tuples
// ((A, a)) ((B, b))...
#define SHARK_BATCH_MAKE_SEQUENCE( TUPLES )\
BOOST_PP_CAT(SHARK_BATCH_ADAPT_STRUCT_FILLER_0 TUPLES, _END)

#define SHARK_BATCH_ADAPT_STRUCT_FILLER_0(X, Y)\
    ((X, Y)) SHARK_BATCH_ADAPT_STRUCT_FILLER_1
#define SHARK_BATCH_ADAPT_STRUCT_FILLER_1(X, Y)\
    ((X, Y)) SHARK_BATCH_ADAPT_STRUCT_FILLER_0
#define SHARK_BATCH_ADAPT_STRUCT_FILLER_0_END
#define SHARK_BATCH_ADAPT_STRUCT_FILLER_1_END


//for a struct S{ A memA; B memB; ...} generates a string of the form
//{memA,memB,memC,memD}
#define SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ)\
{BOOST_PP_SEQ_FOR_EACH_I(\
			SHARK_BATCH_MAKE_AGGREGATE_NAME,\
			~,\
			ATTRIBUTES_SEQ)}
			

#define SHARK_BATCH_MAKE_AGGREGATE_NAME(R, DATA, N, ATTRIBUTE)\
	BOOST_PP_COMMA_IF(N) BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE)			


//generates member list A memA; B memB; ...
#define SHARK_BATCH_DECLARE_MEMBERS(ATTRIBUTES_SEQ) \
BOOST_PP_SEQ_FOR_EACH_I(\
		SHARK_BATCH_MAKE_DATA_MEMBER,\
		~,\
		ATTRIBUTES_SEQ)
		
#define SHARK_BATCH_MAKE_DATA_MEMBER(R, DATA, N, ATTRIBUTE)\
	BOOST_PP_TUPLE_ELEM(2, 0, ATTRIBUTE) BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE);
	
//generates argument list A PREFIXmemA, B PREFIXmemC, C PREFIXmemC,...
#define SHARK_BATCH_DECLARE_ARG_LIST(PREFIX,ATTRIBUTES_SEQ) \
BOOST_PP_SEQ_FOR_EACH_I(\
			SHARK_BATCH_MAKE_ARG_LIST_ELEM,\
			PREFIX,\
			ATTRIBUTES_SEQ)
			

#define SHARK_BATCH_MAKE_ARG_LIST_ELEM(R, PREFIX, N, ATTRIBUTE)\
	BOOST_PP_COMMA_IF(N) BOOST_PP_TUPLE_ELEM(2, 0, ATTRIBUTE)	BOOST_PP_CAT( PREFIX, BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE))

//generates init list memA(PREFIXmemA)
#define SHARK_BATCH_DECLARE_INIT_LIST(PREFIX, ATTRIBUTES_SEQ) \
BOOST_PP_SEQ_FOR_EACH_I(\
			SHARK_BATCH_MAKE_INIT_LIST_ELEM,\
			PREFIX,\
			ATTRIBUTES_SEQ)
			

#define SHARK_BATCH_MAKE_INIT_LIST_ELEM(R, PREFIX, N, ATTRIBUTE)\
	BOOST_PP_COMMA_IF(N) BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE)( BOOST_PP_CAT( PREFIX,  BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE)))

//generates init list memA(PREFIXmemA), where prefix could be some arbitrary prefix like "other."
#define SHARK_BATCH_DECLARE_COPY_INIT_LIST(PREFIX, ATTRIBUTES_SEQ) \
BOOST_PP_SEQ_FOR_EACH_I(\
			SHARK_BATCH_MAKE_COPY_INIT_LIST_ELEM,\
			PREFIX,\
			ATTRIBUTES_SEQ)
			

#define SHARK_BATCH_MAKE_COPY_INIT_LIST_ELEM(R, PREFIX, N, ATTRIBUTE)\
	BOOST_PP_COMMA_IF(N) BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE)( PREFIX .  BOOST_PP_TUPLE_ELEM(2, 1, ATTRIBUTE))


//transformes a list of tuples ((A, a)) ((B, b)) ...
//to ((Batch<A>::TYPE, a)) ((Batch<B>::TYPE, b))...
#define SHARK_BATCH_TRANSFORM_TYPES(TYPE,ATTRIBUTES_SEQ)\
	BOOST_PP_SEQ_TRANSFORM(\
		SHARK_BATCH_TRANSFORM_BATCH_ATTRIBUTES_IMPL,\
		TYPE, ATTRIBUTES_SEQ)
		
#define SHARK_BATCH_TRANSFORM_BATCH_ATTRIBUTES_IMPL(s,TYPE,ELEM)\
	( typename Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::TYPE,BOOST_PP_TUPLE_ELEM(2, 1, ELEM))


//various helper macros...

#define SHARK_BATCH_ASSIGNMENT_HELPER(s,FUNC,N, ELEM)\
	BOOST_PP_TUPLE_ELEM(2, 1, ELEM) = other.BOOST_PP_TUPLE_ELEM(2, 1, ELEM);
	
#define SHARK_BATCH_MOVE_ASSIGNMENT_HELPER(s,FUNC, N,ELEM)\
	BOOST_PP_TUPLE_ELEM(2, 1, ELEM) = std::move(other.BOOST_PP_TUPLE_ELEM(2, 1, ELEM));
	
#define SHARK_BATCH_GET_HELPER(s,FUNC,ELEM)\
	( BOOST_PP_TUPLE_ELEM(2, 0, ELEM), Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::get(batch.BOOST_PP_TUPLE_ELEM(2, 1, ELEM), i))

#define SHARK_BATCH_CREATE_BATCH_HELPER(s,FUNC,ELEM)\
	( BOOST_PP_TUPLE_ELEM(2, 0, ELEM), Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::createBatch(input. BOOST_PP_TUPLE_ELEM(2, 1, ELEM), size))

#define SHARK_BATCH_CREATE_BATCH_FROM_SHAPE_HELPER(s,FUNC,ELEM)\
	( BOOST_PP_TUPLE_ELEM(2, 0, ELEM), Batch<BOOST_PP_TUPLE_ELEM(2, 0, ELEM)>::createBatchFromShape(shape.BOOST_PP_TUPLE_ELEM(2, 1, ELEM), size))

#define SHARK_BATCH_SERIALIZATION_HELPER(s,FUNC,N, ELEM)\
	ar & BOOST_PP_TUPLE_ELEM(2, 1, ELEM);
	
//extension for batches of aggregate type
//this is to handle adapted batches, their references and proxies as a tuple type
//for an adapted struct struct S{ A memA; B memB; ...} and S s;
// we have Batch<S>::tuple_elem<N>(s)
#define SHARK_BATCH_GENERATE_TUPLE_ACCESSORS(ATTRIBUTES_SEQ)\
BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_GENERATE_TUPLE_ACCESSORS_IMPL,  ~, ATTRIBUTES_SEQ);\
typedef std::integral_constant<int,BOOST_PP_SEQ_SIZE(ATTRIBUTES_SEQ)> tuple_size;\
template<std::size_t N, class T>\
static auto tuple_elem(T&& input) -> decltype(get_tuple_elem(std::forward<T>(input), std::integral_constant<int,N>())){\
	return get_tuple_elem(std::forward<T>(input), std::integral_constant<int,N>());\
}

#define SHARK_BATCH_GENERATE_TUPLE_ACCESSORS_IMPL(s, _, N, ELEM)\
template<class T>\
static auto get_tuple_elem(T&& input, std::integral_constant<int,N>) -> decltype((std::forward<T>(input).BOOST_PP_TUPLE_ELEM(2, 1, ELEM)))\
{return std::forward<T>(input).BOOST_PP_TUPLE_ELEM(2, 1, ELEM);}

//the main macro takes inputs of the form ((TypeA, nameA)) ((TypeB, nameB))...
#define SHARK_CREATE_BATCH_INTERFACE_SEQ(ATTRIBUTES_SEQ)\
	struct shape_type{\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(shape_type,ATTRIBUTES_SEQ))\
		template<class Archive>\
		void serialize(Archive & ar, unsigned int const){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_SERIALIZATION_HELPER,  ~, ATTRIBUTES_SEQ);\
		}\
	};\
	struct const_proxy_type{\
		typedef Batch::value_type value_type;\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(const_proxy_type,ATTRIBUTES_SEQ))\
		std::size_t size() const{\
			return batchSize(BOOST_PP_TUPLE_ELEM(2,1,BOOST_PP_SEQ_ELEM(0, ATTRIBUTES_SEQ)));\
		}\
	};\
	struct proxy_type{\
		typedef Batch::value_type value_type;\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(proxy_type,ATTRIBUTES_SEQ))\
		template<class Other>\
		proxy_type& operator=(Other const& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		std::size_t size() const{\
			return batchSize(BOOST_PP_TUPLE_ELEM(2,1,BOOST_PP_SEQ_ELEM(0, ATTRIBUTES_SEQ)));\
		}\
		operator const_proxy_type() const{return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
	};\
	struct type{\
		typedef Batch::value_type value_type;\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(type,ATTRIBUTES_SEQ))\
		type()=default;\
		type(type const&) = default;\
		type(type &&) = default;\
		type(const_proxy_type other):SHARK_BATCH_DECLARE_COPY_INIT_LIST(other , ATTRIBUTES_SEQ){}\
		type(proxy_type other):SHARK_BATCH_DECLARE_COPY_INIT_LIST(other , ATTRIBUTES_SEQ){}\
		type(SHARK_BATCH_DECLARE_ARG_LIST(ARG,SHARK_BATCH_TRANSFORM_TYPES(const_proxy_type,ATTRIBUTES_SEQ) ) ):SHARK_BATCH_DECLARE_INIT_LIST(ARG, ATTRIBUTES_SEQ){}\
		std::size_t size() const{\
			return batchSize(BOOST_PP_TUPLE_ELEM(2,1,BOOST_PP_SEQ_ELEM(0, ATTRIBUTES_SEQ)));\
		}\
		type& operator=(type const&) = default;\
		type& operator=(type &&) = default;\
		template<class Other>\
		type& operator=(Other const& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		operator proxy_type(){return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
		operator const_proxy_type() const{return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
		template<class Archive>\
		void serialize(Archive & ar, unsigned int const){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_SERIALIZATION_HELPER,  ~, ATTRIBUTES_SEQ);\
		}\
	};\
	struct const_reference{\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(const_reference,ATTRIBUTES_SEQ))\
		operator value_type() const{return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
	};\
	struct reference{\
		SHARK_BATCH_DECLARE_MEMBERS(SHARK_BATCH_TRANSFORM_TYPES(reference,ATTRIBUTES_SEQ))\
		reference& operator=(value_type const& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		reference& operator=(value_type&& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_MOVE_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		reference& operator=(reference const& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		reference& operator=(const_reference const& other){\
			BOOST_PP_SEQ_FOR_EACH_I(SHARK_BATCH_ASSIGNMENT_HELPER,  ~, ATTRIBUTES_SEQ);\
			return *this;\
		}\
		operator value_type() const{return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
		operator const_reference() const{return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(ATTRIBUTES_SEQ);}\
	};\
	template<class T>\
	static std::size_t size(T const& batch){return batch.size();}\
	\
	static reference get(type& batch, std::size_t i){\
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_GET_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
	}\
	static const_reference get(type const& batch, std::size_t i){\
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_GET_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
	}\
	static reference get(proxy_type const& batch, std::size_t i){\
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_GET_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
	}\
	static const_reference get(const_proxy_type const& batch, std::size_t i){\
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_GET_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
	}\
	static type createBatch(value_type const& input, std::size_t size = 1){\
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_CREATE_BATCH_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
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
		return SHARK_BATCH_MAKE_AGGREGATE_INITIALIZER(\
			BOOST_PP_SEQ_TRANSFORM(SHARK_BATCH_CREATE_BATCH_FROM_SHAPE_HELPER,  ~, ATTRIBUTES_SEQ)\
		);\
	}\
	SHARK_BATCH_GENERATE_TUPLE_ACCESSORS(ATTRIBUTES_SEQ)

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
///     Batch<T>::type B;
///};
///</code>
///In this case the macro can be used to generate a complete specialisation of Batch<DataType<T> >
///<code>
/// template<class T>
/// SHARK_CREATE_BATCH_INTERFACE( DataType<T>,(RealVector, A)(T, B))
///};
///</code>
#define SHARK_CREATE_BATCH_INTERFACE(NAME,ATTRIBUTES)\
struct Batch< NAME >{\
	typedef NAME value_type;\
	SHARK_CREATE_BATCH_INTERFACE_SEQ(SHARK_BATCH_MAKE_SEQUENCE(ATTRIBUTES))\
};
	
#endif
