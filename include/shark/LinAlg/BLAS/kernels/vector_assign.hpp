#ifndef SHARK_LINALG_BLAS_UBLAS_KERNELS_VECTOR_ASSIGN_HPP
#define SHARK_LINALG_BLAS_UBLAS_KERNELS_VECTOR_ASSIGN_HPP

#include "../functional.hpp"
#include "../expression_types.hpp"

namespace shark {
namespace blas {
namespace kernel {

template<template <class T1, class T2> class F, class V>
void assign(vector_expression<V>& v, typename V::value_type t) {
	 F<typename V::iterator::reference, typename V::value_type> f;
	typedef typename V::iterator iterator;
	iterator end = v().end();
	for (iterator it = v().begin(); it != end; ++it){
		f(*it, t);
	}
}

/////////////////////////////////////////////////////////
//direct assignment of two vectors
////////////////////////////////////////////////////////

// Dense-Dense case
template< class V, class E>
void assign(
	vector_expression<V>& v, vector_expression<E> const& e, 
	dense_random_access_iterator_tag, dense_random_access_iterator_tag
) {
	SIZE_CHECK(v().size() == e().size());
	for(std::size_t i = 0; i != v().size(); ++i){
		v()(i)=e()(i);
	}
}
//Dense-Sparse case
template<class V, class E>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e, 
	dense_random_access_iterator_tag, 
	sparse_bidirectional_iterator_tag
) {
	v().clear();
	typedef typename E::const_iterator iterator;
	iterator end = e().end();
	for(iterator it = e().begin(); it != end; ++it){
		v()(it.index()) = *it;
	}
}
//Sparse-Dense
template<class V, class E>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	sparse_bidirectional_iterator_tag,
	dense_random_access_iterator_tag
) {
	v().clear();
	v().reserve(e().size());
	typename V::iterator targetPos = v().begin();
	RANGE_CHECK(targetPos == v().end());//as v is cleared, pos must be equal to end
	for(std::size_t i = 0; i != e().size(); ++i,++targetPos){
		targetPos = v().set_element(targetPos,i,e()(i));
	}
}
// Sparse-Sparse case
template<class V, class E>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	sparse_bidirectional_iterator_tag,
	sparse_bidirectional_iterator_tag
) {
	v().clear();
	typedef typename E::const_iterator iteratorE;
	typename V::iterator targetPos = v().begin();
	RANGE_CHECK(targetPos == v().end());//as v is cleared, pos must be equal to end
	iteratorE end = e().end();
	for(iteratorE it = e().begin(); it != end; ++it,++targetPos){
		targetPos = v().set_element(targetPos,it.index(),*it);
	}
}

//dispatcher
template< class V, class E>
void assign(vector_expression<V>& v, const vector_expression<E> &e) {
	SIZE_CHECK(v().size() == e().size());
	typedef typename V::const_iterator::iterator_category CategoryV;
	typedef typename E::const_iterator::iterator_category CategoryE;
	assign(v, e, CategoryV(),CategoryE());
}

////////////////////////////////////////////
//assignment with functor
////////////////////////////////////////////

//dense dense case
template<class V, class E, class F>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	F f,
	dense_random_access_iterator_tag, dense_random_access_iterator_tag
) {
	SIZE_CHECK(v().size() == e().size());
	for(std::size_t i = 0; i != v().size(); ++i){
		f(v()(i),e()(i));
	}
}
//Dense-Sparse case
template<class V, class E, class F>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	F f,
	dense_random_access_iterator_tag, sparse_bidirectional_iterator_tag
) {
	typedef typename E::const_iterator iterator;
	iterator end = e().end();
	for(iterator it = e().begin(); it != end; ++it){
		f(v()(it.index()),*it);
	}
}

//sparse-dense case
template<class V, class E, class F>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	F f,
	sparse_bidirectional_iterator_tag tag, dense_random_access_iterator_tag
){	
	typedef typename V::value_type value_type;
	typedef typename V::size_type size_type;
	value_type zero = value_type();
	size_type size = e().size();
	
	typename V::iterator it = v().begin();
	for(size_type i = 0; i != size; ++i,++it){
		if(it == v().end() || it.index() != i){//insert missing elements
			it = v().set_element(it,i,zero); 
		}
		f(*it, e()(i));
	}
}

// Sparse-Sparse case has three implementations.
//the stupidity of this case is, that we have to assume in the general case v and e share the same 
//array memory and thus changing v might invalidate the iterators of e. 
//This is not the same as aliasing of v and e, as v might be for example one matrix row and e another
//of the same matrix.
//thus we look at the cases where (at least) one of the arguments is a vector-container, which means
//that we are not facing the problem of same memory as this would otherwise mean that we are aliasing
//in which case the expression is not defined anyways. 

//called for independent argumeents v and e
template<class V, class E, class F>
void assign_sparse(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	F f
){	
	typedef typename V::value_type value_type;
	typedef typename V::size_type size_type;
	value_type zero = value_type();
	size_type size = v().size();

	typename V::iterator it = v().begin();
	typename E::const_iterator ite = e().begin();
	typename E::const_iterator ite_end = e().end();
	while(it != v().end() && ite != ite_end) {
		size_type it_index = it.index();
		size_type ite_index = ite.index();
		if (it_index == ite_index) {
			f(*it, *ite);
			++ ite;
		} else if (it_index < ite_index) {
			f(*it, zero);
		} else{//it_index > ite_index so insert new element in v()
			it = v().set_element(it,ite_index,zero); 
			f(*it, *ite);
			++ite;
		}
		++it;
	}
	//apply zero transformation on elements which are not transformed yet
	for(;it != v().end();++it) {
		f(*it, zero);
	}
	//add missing elements
	for(;ite != ite_end;++it,++ite) {
		it = v().set_element(it,ite.index(),zero); 
		f(*it, *ite);
	}
}
//as long as one argument is not a proxy, we are in the good case.
template<class V, class E, class F>
void assign(
	vector_expression<V>& v,
	vector_container<E> const& e,
	F f,
	sparse_bidirectional_iterator_tag tag, sparse_bidirectional_iterator_tag
){	
	assign_sparse(v,e);
}
template<class V, class E, class F>
void assign(
	vector_container<V>& v,
	vector_expression<E> const& e,
	F f,
	sparse_bidirectional_iterator_tag tag, sparse_bidirectional_iterator_tag
){	
	assign_sparse(v,e,f);
}
template<class V, class E, class F>
void assign(
	vector_container<V>& v,
	vector_container<E> const& e,
	F f,
	sparse_bidirectional_iterator_tag tag, sparse_bidirectional_iterator_tag
){	
	assign_sparse(v,e,f);
}

//In the general case we have to take one bullet: 
//either use a temporary, which has copying time and allocation time
//or count the non-zero elements of e which might be expensive as well. we decide
//to take the first route for now.
template<class V, class E, class F>
void assign(
	vector_expression<V>& v,
	vector_expression<E> const& e,
	F f,
	sparse_bidirectional_iterator_tag tag, sparse_bidirectional_iterator_tag
){	
	typename vector_temporary<V>::type temporary(v());
	assign_sparse(temporary,e, f);
	v().clear();
	assign(v, temporary);
}

// Dispatcher
template<template <class T1, class T2> class F, class V, class E>
void assign(vector_expression<V>& v, const vector_expression<E> &e) {
	SIZE_CHECK(v().size() == e().size());
	typedef typename V::const_iterator::iterator_category CategoryV;
	typedef typename E::const_iterator::iterator_category CategoryE;
	typedef F<typename V::iterator::reference, typename E::value_type> functor_type;
	assign(v(), e(), functor_type(), CategoryV(),CategoryE());
}

}}}
#endif
