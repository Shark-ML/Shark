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
#ifndef REMORA_VECTOR_EXPRESSION_CLASSES_HPP
#define REMORA_VECTOR_EXPRESSION_CLASSES_HPP

#include "../assignment.hpp"
#include "../proxy_expressions.hpp"
#include "traits.hpp"

namespace remora{

///\brief Implements multiplications of a vector by a scalar
template<class E>
class vector_scalar_multiply:public vector_expression<vector_scalar_multiply <E>, typename E::device_type > {
private:
	typedef typename device_traits<typename E::device_type>:: template multiply_scalar<typename E::value_type> functor_type;
public:
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::value_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;

	typedef vector_scalar_multiply const_closure_type;
	typedef vector_scalar_multiply closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;

	// Construction and destruction
	// May be used as mutable expression.
	vector_scalar_multiply(expression_closure_type const& e, value_type scalar):
		m_expression(e), m_scalar(scalar) {}

	// to_functors
	size_type size() const {
		return m_expression.size();
	}

	// Expression to_functors
	expression_closure_type const& expression() const {
		return m_expression;
	}
	
	value_type scalar()const{
		return m_scalar;
	}
	functor_type functor()const{
		return functor_type(m_scalar);
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// Element access
	const_reference operator()(std::size_t i) const {
		return m_scalar * m_expression(i);
	}
	
	//iterators
	typedef typename device_traits<device_type>:: template transform_iterator<typename expression_closure_type::const_iterator,functor_type >::type const_iterator;
	typedef const_iterator iterator;
	
	const_iterator begin() const {
		return const_iterator(m_expression.begin(),functor_type(m_scalar));
	}
	const_iterator end() const {
		return const_iterator(m_expression.end(),functor_type(m_scalar));
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		auto eval_e = eval_block(m_expression);
		assign(x,vector_scalar_multiply<decltype(eval_e)>(eval_e, m_scalar));
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		auto eval_e = eval_block(m_expression);
		plus_assign(x,vector_scalar_multiply<decltype(eval_e)>(eval_e, m_scalar));
	}
private:
	expression_closure_type m_expression;
	value_type m_scalar;
};

/// \brief Vector expression representing a constant valued vector.
template<class T, class Device>
class scalar_vector:public vector_expression<scalar_vector<T, Device>, Device > {
public:
	typedef std::size_t size_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef scalar_vector const_closure_type;
	typedef scalar_vector closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	
	// Construction and destruction
	scalar_vector()
	:m_size(0), m_value() {}
	explicit scalar_vector(size_type size, value_type value)
	:m_size(size), m_value(value) {}
	scalar_vector(const scalar_vector& v)
	:m_size(v.m_size), m_value(v.m_value) {}

	// to_functors
	size_type size() const {
		return m_size;
	}
    
    T scalar() const {
		return m_value;
	}

	// Element access
	const_reference operator()(std::size_t) const {
		return m_value;
	}
	
	typename device_traits<Device>::queue_type& queue()const{
		return device_traits<Device>::default_queue();
	}
public:
	typedef typename device_traits<Device>:: template constant_iterator<T>::type iterator;
	typedef typename device_traits<Device>:: template constant_iterator<T>::type const_iterator;

	const_iterator begin() const {
		return const_iterator(m_value,0);
	}
	const_iterator end() const {
		return const_iterator(m_value,m_size);
	}

private:
	size_type m_size;
	value_type m_value;
};


/// \brief Vector expression representing the ith unit vector
template<class T, class Device>
class unit_vector:public vector_expression<unit_vector<T, Device>, Device > {
public:
	typedef std::size_t size_type;
	typedef T value_type;
	typedef T const_reference;
	typedef const_reference reference;

	typedef unit_vector const_closure_type;
	typedef unit_vector closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef elementwise<sparse_tag> evaluation_category;
	
	// Construction and destruction
	unit_vector()
	:m_size(0), m_index(0), m_value(1) {}
	explicit unit_vector(size_type size, size_type index, value_type value = value_type(1))
	:m_size(size), m_index(index), m_value(value) {}

	size_type size() const {
		return m_size;
	}

    value_type scalar() const {
		return m_value;
	}
    
    size_type index() const {
		return m_index;
	}
    
	// Element access
	const_reference operator()(size_type const& i) const {
		REMORA_SIZE_CHECK(i < m_size);
		return (i == m_index)? m_value : value_type();
	}
	typename device_traits<Device>::queue_type& queue()const{
		return device_traits<Device>::default_queue();
	}
public:
	typedef typename device_traits<Device>:: template one_hot_iterator<value_type const>::type const_iterator;
	typedef const_iterator iterator;

	const_iterator begin() const {
		return const_iterator(m_index,m_value,false);
	}
	const_iterator end() const {
		return const_iterator(m_index,m_value,true);
	}

private:
	size_type m_size;
	size_type m_index;
    value_type m_value;
};

///\brief Class implementing vector transformation expressions.
///
///transforms a vector Expression e of type E using a Function f of type F as an elementwise transformation f(e(i))
///This transformation needs f to be constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. 
template<class E, class F>
class vector_unary: public vector_expression<vector_unary<E, F>, typename E::device_type > {
public:
	typedef F functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef vector_unary const_closure_type;
	typedef vector_unary closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;
	// Construction and destruction
	// May be used as mutable expression.
	vector_unary(expression_closure_type const& e, F const &functor):
		m_expression(e), m_functor(functor) {}

	// to_functors
	size_type size() const {
		return m_expression.size();
	}

	// Expression to_functors
	expression_closure_type const &expression() const {
		return m_expression;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}	
	
	// Element access
	const_reference operator()(std::size_t i) const {
		return m_functor(m_expression(i));
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		assign(x,m_expression);
		kernels::apply(x,m_functor);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		auto eval_rhs = eval_block(m_expression);
		//merge the multiplication with the functor to run only one kernel.
		//also make use of that we can now run the assignment kernel directly with that functor
		typedef typename device_traits<device_type>::template add<typename VecX::value_type> Add;
		typedef typename device_traits<device_type>::template identity<typename VecX::value_type> Identity;
		typedef typename device_traits<device_type>::template transform_arguments< Identity, F, Add> Composed;
		kernels::assign(x,eval_rhs,Composed(Identity(), m_functor,Add()));
	}

	
	typedef typename device_traits<device_type>:: template transform_iterator<
		typename expression_closure_type::const_iterator,
		functor_type
	>::type const_iterator;
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
class vector_addition: public vector_expression<vector_addition<E1,E2>, typename E1::device_type > {
private:
	typedef typename device_traits<typename E1::device_type>:: template add<typename E1::value_type> functor_type;
public:
	typedef typename common_value_type<E1,E2>::type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef typename E1::size_type size_type;

	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
	
	typedef vector_addition const_closure_type;
	typedef vector_addition closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;

	// Construction and destruction
	explicit vector_addition (
		lhs_closure_type e1, 
		rhs_closure_type e2
	):m_lhs(e1),m_rhs(e2){
		REMORA_SIZE_CHECK(e1.size() == e2.size());
	}

	// to_functors
	size_type size() const {
		return m_lhs.size();
	}

	// Expression to_functors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	// Element access
	const_reference operator()(std::size_t i) const {
		return m_lhs(i) + m_rhs(i);
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		//working around a bug in non-dense assign
		if(!std::is_base_of<dense_tag, typename E1::evaluation_category::tag>::value){
			x().clear();
			plus_assign(x, m_lhs);
		}else{
			assign(x, m_lhs);
		}
		plus_assign(x,m_rhs);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		plus_assign(x,m_lhs);
		plus_assign(x,m_rhs);
	}

	// Iterator types
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename lhs_closure_type::const_iterator,
		typename rhs_closure_type::const_iterator,
		functor_type
	>::type const_iterator;
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
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};

template<class E1, class E2, class F>
class vector_binary:public vector_expression<vector_binary<E1,E2, F>,typename E1::device_type > {
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
public:
	typedef F functor_type;
	typedef typename F::result_type value_type;
	typedef typename E1::size_type size_type;
	typedef value_type const_reference;
	typedef value_type reference;
	
	typedef vector_binary const_closure_type;
	typedef vector_binary closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;
	// Construction and destruction
	explicit vector_binary (
		lhs_closure_type e1, 
		rhs_closure_type e2,
		F functor
	):m_lhs(e1),m_rhs(e2), m_functor(functor) {
		REMORA_SIZE_CHECK(e1.size() == e2.size());
	}

	// to_functors
	size_type size() const {
		return m_lhs.size ();
	}

	// Expression to_functors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}
	
	functor_type const& functor()const{
		return m_functor;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	// Element access
	const_reference operator()(std::size_t i) const {
		return m_functor(m_lhs(i), m_rhs(i));
	}
	
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		assign(x,m_lhs);
		auto eval_rhs = eval_block(m_rhs);
		kernels::assign(x,eval_rhs,m_functor);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		auto eval_lhs = eval_block(m_lhs);
		auto eval_rhs = eval_block(m_rhs);
		vector_binary<decltype(eval_lhs),decltype(eval_rhs),F> e(eval_lhs,eval_rhs, m_functor);
		plus_assign(x,e);	
	}

	// Iterator types
	
	// Iterator enhances the iterator of the referenced expressions
	// with the unary functor.
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename lhs_closure_type::const_iterator,
		typename rhs_closure_type::const_iterator,
		functor_type
	>::type const_iterator;
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
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
	functor_type m_functor;
};


//given vectors v and w, forms vector (v,w)
template<class E1, class E2>
class vector_concat: public vector_expression<vector_concat<E1,E2>, typename E1::device_type > {
private:
	typedef typename device_traits<typename E1::device_type>:: template add<typename E1::value_type> functor_type;
public:
	typedef typename common_value_type<E1,E2>::type value_type;
	typedef value_type const_reference;
	typedef value_type reference;
	typedef typename E1::size_type size_type;

	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
	
	typedef vector_concat const_closure_type;
	typedef vector_concat closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename evaluation_tag_restrict_traits<
		typename E1::evaluation_category::tag,
		typename E2::evaluation_category::tag
	>::type> evaluation_category;
	typedef typename E1::device_type device_type;

	// Construction and destruction
	explicit vector_concat(
		lhs_closure_type e1, 
		rhs_closure_type e2
	):m_lhs(e1),m_rhs(e2){}

	// to_functors
	size_type size() const {
		return m_lhs.size() + m_rhs.size();
	}

	// Expression to_functors
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		auto left = subrange(x,0,m_lhs.size()); 
		auto right = subrange(x,m_lhs.size(),x().size()); 
		assign(left,m_lhs);
		assign(right,m_rhs);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		auto left = subrange(x,0,m_lhs.size()); 
		auto right = subrange(x,m_lhs.size(),x().size()); 
		plus_assign(left,m_lhs);
		plus_assign(right,m_rhs);
	}

	// Iterator types
	typedef no_iterator const_iterator;
	typedef no_iterator iterator;

private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};


//traits for transforming an expression into a functor for gpu usage

template<class E>
struct ExpressionToFunctor<vector_scalar_multiply<E> >{
	typedef typename E::device_type device_type;
	typedef typename device_traits<device_type>::template multiply_scalar<typename E::value_type> functor_type;
	static auto transform(vector_scalar_multiply<E> const& e) -> decltype(device_traits<device_type>::make_compose(to_functor(e.expression()),e.functor())){
		return device_traits<device_type>::make_compose(to_functor(e.expression()), e.functor());
	}
};
template<class E1, class E2>
struct ExpressionToFunctor<vector_addition<E1, E2> >{
	typedef typename E1::device_type device_type;
	typedef typename device_traits<device_type>::template add<typename vector_addition<E1, E2>::value_type> functor_type;
	static auto transform(vector_addition<E1, E2> const& e) -> decltype(device_traits<device_type>::make_compose_binary(to_functor(e.lhs()),to_functor(e.rhs()),functor_type())){
		return device_traits<device_type>::make_compose_binary(to_functor(e.lhs()),to_functor(e.rhs()),functor_type());
	}
};

template<class E, class F>
struct ExpressionToFunctor<vector_unary<E, F> >{
	typedef typename E::device_type device_type;
	static auto transform(vector_unary<E, F> const& e) -> decltype(device_traits<device_type>::make_compose(to_functor(e.expression()),e.functor())){
		return device_traits<device_type>::make_compose(to_functor(e.expression()),e.functor());
	}
};

template<class E1, class E2, class F>
struct ExpressionToFunctor<vector_binary<E1, E2, F> >{
	typedef typename E1::device_type device_type;
	static auto transform(vector_binary<E1, E2, F> const& e) -> decltype(device_traits<device_type>::make_compose_binary(to_functor(e.lhs()),to_functor(e.rhs()), e.functor())){
		return device_traits<device_type>::make_compose_binary(to_functor(e.lhs()),to_functor(e.rhs()), e.functor());
	}
};

template<class T, class Device>
struct ExpressionToFunctor<scalar_vector<T, Device> >{
	static typename device_traits<Device>::template constant<T> transform(scalar_vector<T, Device> const& e){
		return typename device_traits<Device>::template constant<T>(e.scalar());
	}
};



}
#endif