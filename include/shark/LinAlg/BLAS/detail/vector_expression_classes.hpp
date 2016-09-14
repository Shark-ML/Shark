/*!
 * \brief       Classes used for vector expressions
 * 
 * \author      O. Krause
 * \date        2016
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
 #ifndef SHARK_LINALG_BLAS_VECTOR_EXPRESSION_CLASSES_HPP
#define SHARK_LINALG_BLAS_VECTOR_EXPRESSION_CLASSES_HPP

#include "../assignment.hpp"
#include "iterator.hpp"
#include "traits.hpp"

namespace shark {
namespace blas {

///\brief Implements multiplications of a vector by a scalar
template<class E>
class vector_scalar_multiply:
	public vector_expression<vector_scalar_multiply <E> > {
	typedef vector_scalar_multiply<E> self_type;
public:
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename E::value_type value_type;
	typedef typename E::scalar_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	// May be used as mutable expression.
	vector_scalar_multiply(expression_closure_type const& e, scalar_type scalar):
		m_expression(e), m_scalar(scalar) {}

	// Accessors
	size_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	scalar_type scalar()const{
		return m_scalar;
	}

public:
	// Element access
	const_reference operator()(index_type i) const {
		return m_scalar * m_expression(i);
	}

	const_reference operator[](index_type i) const {
		return m_scalar * m_expression(i);
	}
	
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.assign_to(x,alpha*m_scalar);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.plus_assign_to(x,alpha*m_scalar);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		m_expression.minus_assign_to(x,alpha*m_scalar);
	}

	
	//iterators
	typedef transform_iterator<typename E::const_iterator,scalar_multiply1<value_type, scalar_type> > const_iterator;
	typedef const_iterator iterator;
	
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),scalar_multiply1<value_type, scalar_type>(m_scalar));
	}
private:
	expression_closure_type m_expression;
	scalar_type m_scalar;
};

/// \brief Vector expression representing a constant valued vector.
template<class T>
class scalar_vector:public vector_expression<scalar_vector<T> > {

	typedef scalar_vector<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef T scalar_type;
	typedef const T& const_reference;
	typedef const_reference reference;
	typedef T const* const_pointer;
	typedef const_pointer pointer;
	

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef elementwise_tag evaluation_category;

	// Construction and destruction
	scalar_vector()
	:m_size(0), m_value() {}
	explicit scalar_vector(size_type size, value_type value)
	:m_size(size), m_value(value) {}
	scalar_vector(const scalar_vector& v)
	:m_size(v.m_size), m_value(v.m_value) {}

	// Accessors
	size_type size() const {
		return m_size;
	}

	// Element access
	const_reference operator()(index_type /*i*/) const {
		return m_value;
	}

	const_reference operator [](index_type /*i*/) const {
		return m_value;
	}

public:
	typedef constant_iterator<T> iterator;
	typedef constant_iterator<T> const_iterator;

	const_iterator begin() const {
		return const_iterator(0,m_value);
	}
	const_iterator end() const {
		return const_iterator(m_size,m_value);
	}

private:
	size_type m_size;
	value_type m_value;
};

///\brief Class implementing vector transformation expressions.
///
///transforms a vector Expression e of type E using a Function f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
template<class E, class F>
class vector_unary:
	public vector_expression<vector_unary<E, F> > {
	typedef vector_unary<E, F> self_type;
public:
	typedef F functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	// May be used as mutable expression.
	vector_unary(expression_closure_type const& e, F const &functor):
		m_expression(e), m_functor(functor) {}

	// Accessors
	size_type size() const {
		return m_expression.size();
	}

	// Expression accessors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//compute this by first assigning the result of the argument and then applying
		//the function to every element
		assign(x,m_expression);
		typename VecX::iterator end=x().end();
		for(typename VecX::iterator pos =x().begin(); pos != end; ++pos){
			*pos= alpha * m_functor(*pos);
		}
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//First assign result of this expression to a temporary and then perform plus_assignment to x
		typename vector_temporary<self_type>::type temporary(size());
		assign_to(temporary,alpha);
		plus_assign_to(x,temporary);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		//First assign result of this expression to a temporary and then perform minus_assignment to x
		typename vector_temporary<self_type>::type temporary(size());
		assign_to(temporary,alpha);
		minus_assign_to(x,temporary);
	}

public:
	// Element access
	const_reference operator()(index_type i) const {
		return m_functor(m_expression(i));
	}

	const_reference operator[](index_type i) const {
		return m_functor(m_expression[i]);
	}

	typedef transform_iterator<typename E::const_iterator,functor_type> const_iterator;
	typedef const_iterator iterator;

	// Element lookup
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),m_functor);
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),m_functor);
	}
private:
	expression_closure_type m_expression;
	F m_functor;
};

template<class E1, class E2>
class vector_addition: public vector_expression<vector_addition<E1,E2> > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	
	typedef vector_addition const_closure_type;
	typedef vector_addition closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	explicit vector_addition (
		expression_closure1_type e1, 
		expression_closure2_type e2
	):m_lhs(e1),m_rhs(e2){
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	size_type size() const {
		return m_lhs.size();
	}

	// Expression accessors
	expression_closure1_type const& lhs() const {
		return m_lhs;
	}
	expression_closure2_type const& rhs() const {
		return m_rhs;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_lhs(i) + m_rhs(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		assign(x,alpha*m_lhs);
		plus_assign(x,alpha*m_rhs);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign(x,alpha*m_lhs);
		plus_assign(x,alpha*m_rhs);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		minus_assign(x,alpha*m_lhs);
		minus_assign(x,alpha*m_rhs);
	}

	// Iterator types
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator(functor_type(),
			m_lhs.begin(),m_lhs.end(),
			m_rhs.begin(),m_rhs.end()
		);
	}
	const_iterator end() const {
		return const_iterator(functor_type(),
			m_lhs.end(),m_lhs.end(),
			m_rhs.end(),m_rhs.end()
		);
	}

private:
	expression_closure1_type m_lhs;
	expression_closure2_type m_rhs;
};

template<class E1, class E2, class F>
class vector_binary:
	public vector_expression<vector_binary<E1,E2, F> > {
	typedef vector_binary<E1,E2, F> self_type;
	typedef E1 const lhs_type;
	typedef E2 const rhs_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef value_type const * const_pointer;
	typedef value_type const*  pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef F functor_type;
	typedef typename E1::const_closure_type expression_closure1_type;
	typedef typename E2::const_closure_type expression_closure2_type;
	
	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	explicit vector_binary (
		expression_closure1_type e1, 
		expression_closure2_type e2,
		F functor
	):m_lhs(e1),m_rhs(e2), m_functor(functor) {
		SIZE_CHECK(e1.size() == e2.size());
	}

	// Accessors
	size_type size() const {
		return m_lhs.size ();
	}

	// Expression accessors
	expression_closure1_type const& lhs() const {
		return m_lhs;
	}
	expression_closure2_type const& rhs() const {
		return m_rhs;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}

	// Element access
	const_reference operator() (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_lhs(i),m_rhs(i));
	}

	const_reference operator[] (index_type i) const {
		SIZE_CHECK(i < size());
		return m_functor(m_lhs(i),m_rhs(i));
	}
	
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		x().clear();
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		plus_assign_to(x,eval_block(m_lhs), eval_block(m_rhs), -alpha);
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expressions
	// with the unary functor.
	typedef binary_transform_iterator<
		typename E1::const_iterator,
		typename E2::const_iterator,
		functor_type
	> const_iterator;
	typedef const_iterator iterator;

	const_iterator begin () const {
		return const_iterator (m_functor,
			m_lhs.begin(),m_lhs.end(),
			m_rhs.begin(),m_rhs.end()
		);
	}
	const_iterator end() const {
		return const_iterator (m_functor,
			m_lhs.end(),m_lhs.end(),
			m_rhs.end(),m_rhs.end()
		);
	}

private:
	expression_closure1_type m_lhs;
	expression_closure2_type m_rhs;
	F m_functor;

	template<class VecX, class LHS, class RHS>
	void plus_assign_to(
		vector_expression<VecX>& x,
		vector_expression<LHS> const& lhs, vector_expression<RHS> const& rhs,
		scalar_type alpha
	)const{
		//we know that lhs and rhs are elementwise expressions so we can now create the elementwise expression and assign it.
		vector_binary<LHS,RHS, F> e(lhs(),rhs(), m_functor);
		vector_scalar_multiply<vector_binary<LHS,RHS,F> > e1(e,alpha);
		plus_assign(x,e);
	}
};

}}
#endif