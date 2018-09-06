/*!
 * \brief       Classes used for matrix expressions.
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
 #ifndef REMORA_MATRIX_EXPRESSION_CLASSES_HPP
#define REMORA_MATRIX_EXPRESSION_CLASSES_HPP

#include "traits.hpp"
#include "../kernels/gemv.hpp"
#include "../kernels/tpmv.hpp"
#include "../kernels/trmv.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/trmm.hpp"
#include "../kernels/fold_rows.hpp"
#include "../assignment.hpp"
#include "../proxy_expressions.hpp"
#include <type_traits>

namespace remora{

template<class E>
class matrix_scalar_multiply:public matrix_expression<matrix_scalar_multiply<E>, typename E::device_type >{
public:
	typedef typename device_traits<typename E::device_type>::template multiply_scalar<typename E::value_type> functor_type;
	typedef typename E::const_closure_type expression_closure_type;
	typedef typename E::size_type size_type;
	typedef typename E::value_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::orientation orientation;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;

	// Construction
	explicit matrix_scalar_multiply(expression_closure_type const& e, value_type scalar)
	:m_expression(e), m_scalar(scalar){}

	//Accessors 
	size_type size1() const{
		return m_expression.size1();
	}
	size_type size2() const{
		return m_expression.size2();
	}
	value_type scalar()const{
		return m_scalar;
	}
	expression_closure_type const& expression() const{
		return m_expression;
	};
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}

	// Element Functor
	auto elements() const -> decltype(
		device_traits<device_type>::make_compose(
			std::declval<expression_closure_type const&>().elements(), std::declval<functor_type&>()
		)
	){
		return device_traits<device_type>::make_compose(
			m_expression.elements(), functor_type(m_scalar)
		);
	}

	// Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		typename device_traits<device_type>::template multiply_assign<value_type> f(m_scalar);
		kernels::assign(X, eval_e, f);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		auto eval_e = eval_block(m_expression);
		typename device_traits<device_type>::template multiply_and_add<value_type> f(m_scalar);
		kernels::assign(X, eval_e, f);
	}
	
	// Iterator Access
	typedef typename device_traits<device_type>:: template transform_iterator<
		typename expression_closure_type::const_major_iterator, functor_type
	>::type const_major_iterator;
	typedef const_major_iterator major_iterator;
	
	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator(m_expression.major_begin(i),functor_type(m_scalar));
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator(m_expression.major_end(i),functor_type(m_scalar));
	}

private:
	expression_closure_type m_expression;
	value_type m_scalar;
};	
	
template<class E1, class E2>
class matrix_addition: public matrix_expression<matrix_addition<E1, E2>, typename E1::device_type >{
private:
	typedef typename device_traits<typename E1::device_type>:: template add<typename E1::value_type> functor_type;
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;
public:

	typedef typename E1::size_type size_type;
	typedef decltype(typename E1::value_type() + typename E2::value_type()) value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_addition const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename std::conditional< //the orientation is only known if E1 and E2 have the same
		std::is_same<typename E1::orientation, typename E2::orientation>::value,
		typename E1::orientation,
		unknown_orientation
	>::type orientation;
	//the evaluation category is blockwise if one of the expressions is blockwise or
	// if the orientation is unknown (efficient for expressions like A=B+C^T
	typedef typename std::conditional<
		std::is_same<orientation, unknown_orientation>::value,
		blockwise<typename evaluation_restrict_traits<E1,E2>::type::tag>,
		typename evaluation_restrict_traits<E1,E2>::type
	>::type evaluation_category;
	typedef typename E1::device_type device_type;

        // Construction
        explicit matrix_addition(
		lhs_closure_type const& e1,
		rhs_closure_type const& e2
	): m_lhs (e1), m_rhs (e2){}
	
	//Accessors
	size_type size1() const{
		return m_lhs.size1();
        }
	size_type size2() const{
		return m_lhs.size2();
        }
	lhs_closure_type const& lhs()const{
		return m_lhs;
	}
	rhs_closure_type const& rhs()const{
		return m_rhs;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	// Element Functor
	auto elements() const -> decltype(
		device_traits<device_type>::make_compose_binary(
			std::declval<lhs_closure_type const&>().elements(),
			std::declval<rhs_closure_type const&>().elements(),
			std::declval<functor_type&>()
		)
	){
		return device_traits<device_type>::make_compose_binary(
			m_lhs.elements(), m_rhs.elements(), functor_type()
		);
	}

	//Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		//working around a bug in non-dense assign
		if(!std::is_base_of<dense_tag, typename E1::evaluation_category::tag>::value){
			X().clear();
			plus_assign(X, m_lhs);
		}else{
			assign(X, m_lhs);
		}
		plus_assign(X, m_rhs);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		plus_assign(X,m_lhs);
		plus_assign(X,m_rhs);
	}

	//Iterator Access
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename lhs_closure_type::const_major_iterator,
		typename rhs_closure_type::const_major_iterator,
		functor_type
	>::type const_major_iterator;
	typedef const_major_iterator major_iterator;

	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator (functor_type(),
			m_lhs.major_begin(i),m_lhs.major_end(i),
			m_rhs.major_begin(i),m_rhs.major_end(i)
		);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator (functor_type(),
			m_lhs.major_end(i),m_lhs.major_end(i),
			m_rhs.major_end(i),m_rhs.major_end(i)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

template<class V, class Orientation=row_major>
class vector_repeater:public matrix_expression<vector_repeater<V, Orientation>, typename V::device_type >{
public:
	typedef typename V::const_closure_type expression_closure_type;
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef vector_repeater const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef Orientation orientation;
	typedef typename V::evaluation_category evaluation_category;
	typedef typename V::device_type device_type;

	// Construction
	explicit vector_repeater(expression_closure_type const& e, size_type elements):
	m_vector(e), m_elements(elements){}

	//Accessors
	size_type size1() const{
		return orientation::index_M(m_elements,m_vector.size());
	}
	size_type size2() const{
		return orientation::index_m(m_elements,m_vector.size());
	}
	const expression_closure_type& expression() const{
		return m_vector;
	}
	std::size_t num_repetitions()const{
		return m_elements;
	}
	
	// Element Functor
private:
	typedef typename std::conditional<
		std::is_same<orientation, row_major>::value,
		typename device_traits<device_type>::template right_arg<std::size_t>,
		typename device_traits<device_type>::template left_arg<std::size_t>
	>::type argument_functor;
public:
	auto elements() const -> decltype(
		device_traits<device_type>::make_compose(argument_functor(), std::declval<expression_closure_type const&>().elements())
	){
		return device_traits<device_type>::make_compose(argument_functor(), m_vector.elements());
	}

	//Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X) const{
		X().clear();
		plus_assign_to(X,eval_block(m_vector));
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X) const{
		plus_assign_to(X,eval_block(m_vector));
	}
	
	typename device_traits<typename V::device_type>::queue_type& queue()const{
		return m_vector.queue();
	}
	
	//Iterator Access
private:
	typedef typename V::const_iterator vector_iterator;
	typedef typename device_traits<typename V::device_type>:: template constant_iterator<value_type>::type constant_iterator;
	
	vector_iterator begin(size_type i, row_major) const{
		return m_vector.begin();
	}
	vector_iterator end(size_type i, row_major) const{
		return m_vector.end();
	}
	constant_iterator begin(size_type i, column_major) const{
		return constant_iterator(m_vector(i),0);
	}
	constant_iterator end(size_type i, column_major) const{
		return constant_iterator(m_vector(i), m_elements);
	}
public:
	typedef typename std::conditional<std::is_same<Orientation, row_major>::value, vector_iterator, constant_iterator>::type const_major_iterator;
	typedef const_major_iterator major_iterator;


	const_major_iterator major_begin(size_type i) const{
		REMORA_RANGE_CHECK( i < size1());
		return begin(i, orientation());
	}
	const_major_iterator major_end(size_type i) const{
		REMORA_RANGE_CHECK( i < size1());
		return end(i, orientation());
	}
private:
	
	template<class MatX, class VecV>
	void plus_assign_to(
		matrix_expression<MatX, device_type>& X,
		vector_expression<VecV, device_type> const& v
	)const{
		vector_repeater<VecV, Orientation> e(v(),m_elements);
		plus_assign(X,e);
	}

	expression_closure_type m_vector;
	size_type m_elements;
};

/// \brief A matrix with all values of type \c T equal to the same value
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
template<class T, class Device, class Orientation>
class scalar_matrix:public matrix_expression<scalar_matrix<T, Device, Orientation>, Device >{
public:
	typedef std::size_t size_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef scalar_matrix const_closure_type;
	typedef scalar_matrix closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef Orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction
	scalar_matrix(size_type size1, size_type size2, const value_type& value = value_type(1)):
		m_size1(size1), m_size2(size2), m_value(value){}
	
	//Accessors
	size_type size1() const{
		return m_size1;
	}
	size_type size2() const{
		return m_size2;
	}
	T scalar() const{
		return m_value;
	}
	typename device_traits<Device>::queue_type& queue()const{
		return device_traits<Device>::default_queue();
	}
	
	// Element Functor
	typename device_traits<Device>:: template constant<value_type> elements() const{
		return {m_value};
	}
    
	//Iterators
	typedef typename device_traits<Device>:: template constant_iterator<value_type>::type const_major_iterator;
	typedef const_major_iterator major_iterator;

	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator(m_value, 0);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator(m_value, orientation::index_m(size1(),size2()));
	}
private:
	size_type m_size1;
	size_type m_size2;
	value_type m_value;
};

///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Function f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application.
///F must provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
template<class E, class F>
class matrix_unary:public matrix_expression<matrix_unary<E, F>, typename E::device_type >{
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef typename E::size_type size_type;

	typedef matrix_unary const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename E::orientation orientation;
	typedef typename E::evaluation_category evaluation_category;
	typedef typename E::device_type device_type;

	// Construction
	explicit matrix_unary(expression_closure_type const& e, functor_type const& functor):
		m_expression(e), m_functor(functor){}
		
	// Accessors
	size_type size1() const{
		return m_expression.size1();
	}
	size_type size2() const{
		return m_expression.size2();
	}
	expression_closure_type const& expression() const{
		return m_expression;
	}
	functor_type const& functor() const{
		return m_functor;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	// Element Functor
	auto elements() const -> decltype(
		device_traits<device_type>::make_compose(
			std::declval<expression_closure_type const&>().elements(),
			std::declval<functor_type&>()
		)
	){
		return device_traits<device_type>::make_compose(m_expression.elements(), m_functor);
	}

	//Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X) const{
		assign(X,m_expression);
		kernels::apply(X,m_functor);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X) const{
		
		//assign expects functions b=f(b,a)
		//for b=b+g(a)
		//we implement b=b+g(a) = add(identity(b),g(a))
		auto eval_rhs = eval_block(m_expression);
		typename device_traits<device_type>:: template identity<value_type> identity;
		typename device_traits<device_type>:: template add<value_type> add;
		kernels::assign(X,eval_rhs, device_traits<device_type>::make_transform_arguments(identity,m_functor,add));
	}

	// Iterator Access
	typedef typename device_traits<device_type>:: template transform_iterator<typename expression_closure_type::const_major_iterator, F>::type const_major_iterator;
	typedef const_major_iterator major_iterator;
	
	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator(m_expression.major_begin(i),m_functor);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator(m_expression.major_end(i),m_functor);
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;
};

template<class E1, class E2, class F>
class matrix_binary:public matrix_expression<matrix_binary<E1, E2, F>, typename E1::device_type >{
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename F::result_type value_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_binary const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename std::conditional< //the orientation is only known if E1 and E2 have the same
		std::is_same<typename E1::orientation, typename E2::orientation>::value,
		typename E1::orientation,
		unknown_orientation
	>::type orientation;
	typedef typename std::conditional<
		std::is_same<orientation, unknown_orientation>::value,
		blockwise<typename evaluation_restrict_traits<E1,E2>::type::tag>,
		typename evaluation_restrict_traits<E1,E2>::type
	>::type evaluation_category;
	typedef typename E1::device_type device_type;

	typedef F functor_type;

        // Construction
        explicit matrix_binary (
		lhs_closure_type const& e1,  rhs_closure_type const& e2, functor_type functor 
	): m_lhs (e1), m_rhs (e2),m_functor(functor){}
	
	//Accessors
	size_type size1() const{
		return m_lhs.size1();
        }
	size_type size2() const{
		return m_lhs.size2();
        }
	lhs_closure_type const& lhs() const{
		return m_lhs;
	}
	rhs_closure_type const& rhs() const{
		return m_rhs;
	}
	functor_type const& functor() const{
		return m_functor;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	// Element Functor
	auto elements() const -> decltype(
		device_traits<device_type>::make_compose_binary(
			std::declval<E1 const&>().elements(),
			std::declval<E2 const&>().elements(),
			std::declval<functor_type&>()
		)
	){
		return device_traits<device_type>::make_compose_binary(
			m_lhs.elements(), m_rhs.elements(), m_functor
		);
	}
	
	//Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		assign(X,m_lhs);
		kernels::assign(X,eval_block(m_rhs),m_functor);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		auto eval_lhs = eval_block(m_lhs);
		auto eval_rhs = eval_block(m_rhs);
		matrix_binary<decltype(eval_lhs),decltype(eval_rhs),F> e(eval_lhs,eval_rhs, m_functor);
		plus_assign(X,e);		
	}
	
	//Iterator Access
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename lhs_closure_type::const_major_iterator,typename rhs_closure_type::const_major_iterator,functor_type
	>::type const_major_iterator;
	typedef const_major_iterator major_iterator;

	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator (m_functor,
			m_lhs.major_begin(i),m_lhs.major_end(i),
			m_rhs.major_begin(i),m_rhs.major_end(i)
		);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator (m_functor,
			m_lhs.major_end(i),m_lhs.major_end(i),
			m_rhs.major_end(i),m_rhs.major_end(i)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

template<class E1, class E2>
class outer_product:public matrix_expression<outer_product<E1, E2>, typename E1::device_type >{
	typedef typename common_value_type<E1,E2>::type common_arg;
	typedef typename device_traits< typename E1::device_type>:: template multiply_scalar<common_arg> functor_type;
	typedef typename device_traits< typename E1::device_type>:: template multiply<common_arg> functor_type_op;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	
	typedef common_arg value_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef typename E1::size_type size_type;

	typedef outer_product const_closure_type;
	typedef outer_product closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef row_major orientation;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;


	// Construction	
	explicit outer_product(lhs_closure_type const& e1, rhs_closure_type const& e2)
	:m_lhs(e1), m_rhs(e2){}
	
	// Accessors
	size_type size1() const{
		return m_lhs.size();
	}
	size_type size2() const{
		return m_rhs.size();
	}
	lhs_closure_type const& lhs() const{
		return m_lhs;
	}
	rhs_closure_type const& rhs() const{
		return m_rhs;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	//Element Functor
	auto elements() const -> decltype(
		device_traits<device_type>::make_transform_arguments(
			std::declval<E1 const&>().elements(),
			std::declval<E2 const&>().elements(),
			std::declval<functor_type_op&>()
		)
	){
		return device_traits<device_type>::make_transform_arguments(
			m_lhs.elements(), m_rhs.elements(), functor_type_op()
		);
	}
	
	// Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		auto lhs_eval = eval_block(m_lhs);
		auto rhs_eval = eval_block(m_rhs);
		outer_product<decltype(lhs_eval), decltype(rhs_eval)> e(lhs_eval,rhs_eval); 
		assign(X, e);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		auto lhs_eval = eval_block(m_lhs);
		auto rhs_eval = eval_block(m_rhs);
		outer_product<decltype(lhs_eval), decltype(rhs_eval)> e(lhs_eval,rhs_eval); 
		plus_assign(X, e);
	}
	
	// Iterator Access
	typedef typename device_traits<device_type>:: template transform_iterator<typename rhs_closure_type::const_iterator,functor_type>::type const_major_iterator;
	typedef const_major_iterator major_iterator;
	
	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator(m_rhs.begin(),
			functor_type(m_lhs.elements()(i))
		);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator(m_rhs.end(),
			functor_type(m_lhs.elements()(i))
		);
	}
private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};


/// \brief An diagonal matrix with values stored inside a diagonal vector
///
/// the matrix stores a Vector representing the diagonal.
template<class V>
class diagonal_matrix: public matrix_expression<diagonal_matrix< V >, typename V::device_type >{
	typedef typename V::const_closure_type vector_closure_type;
public:
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef value_type const_reference;
	typedef value_type reference;

	typedef diagonal_matrix const_closure_type;
	typedef diagonal_matrix closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef row_major orientation;
	typedef typename V::device_type device_type;

	// Construction
	diagonal_matrix(vector_closure_type const& diagonal):m_diagonal(diagonal){}

	// Accessors
	size_type size1() const{
		return m_diagonal.size();
	}
	size_type size2() const{
		return m_diagonal.size();
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_diagonal.queue();
	}
	vector_closure_type const& expression()const{
		return m_diagonal;
	}
	
	// Element access
	typename device_traits< device_type >:: template diag<
		decltype(std::declval<vector_closure_type>().elements())
	>
	elements() const{
		return {m_diagonal.elements()};
	}
	
	// Iterators
	typedef typename device_traits<device_type>:: template one_hot_iterator<value_type const>::type const_major_iterator;
	typedef const_major_iterator major_iterator;

	const_major_iterator major_begin(size_type i) const{
		return const_major_iterator(i, m_diagonal.elements()(i),false);
	}
	const_major_iterator major_end(size_type i) const{
		return const_major_iterator(i, value_type(),true);
	}

private:
	vector_closure_type m_diagonal; 
};

template<class MatA, class VecV>
class matrix_vector_prod:public vector_expression<matrix_vector_prod<MatA, VecV>, typename MatA::device_type >{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename VecV::const_closure_type vector_closure_type;
public:
	typedef decltype(
		typename MatA::value_type() * typename VecV::value_type()
	) value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_vector_prod<MatA, VecV> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename evaluation_tag_restrict_traits<
		typename MatA::evaluation_category::tag,
		typename VecV::evaluation_category::tag
	>::type> evaluation_category;
	typedef typename MatA::device_type device_type;

	// Construction
	explicit matrix_vector_prod(
		matrix_closure_type const& matrix,
		vector_closure_type const& vector,
		value_type const& alpha
	):m_matrix(matrix), m_vector(vector), m_alpha(alpha){}

	
	// Accessors 
	size_type size() const{
		return m_matrix.size1();
	}
	matrix_closure_type const& matrix() const{
		return m_matrix;
	}
	vector_closure_type const& vector() const{
		return m_vector;
	}
	value_type alpha() const{
		return m_alpha;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		assign_to(x, typename MatA::orientation(), typename MatA::evaluation_category::tag());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		plus_assign_to(x, typename MatA::orientation(), typename MatA::evaluation_category::tag());
	}
	
	// Iterator Access 
	typedef no_iterator const_iterator;
	typedef no_iterator iterator;
	
private:
	//gemv
	template<class VecX, class Category>
	void assign_to(vector_expression<VecX, device_type>& x, linear_structure, Category c)const{
		x().clear();
		plus_assign_to(x, linear_structure(), c);
	}
	template<class VecX, class Category>
	void plus_assign_to(vector_expression<VecX, device_type>& x, linear_structure, Category )const{
		kernels::gemv(eval_block(m_matrix), eval_block(m_vector), x, m_alpha);
	}
	//tpmv
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, triangular_structure, packed_tag )const{
		noalias(x) = eval_block(m_alpha * m_vector);
		kernels::tpmv(eval_block(m_matrix), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, triangular_structure, packed_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp( eval_block(m_alpha * m_vector));
		kernels::tpmv(eval_block(m_matrix), temp);
		noalias(x) += temp;
	}
	//trmv
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, triangular_structure, dense_tag )const{
		noalias(x) = eval_block(m_alpha * m_vector);
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.to_dense(), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, triangular_structure, dense_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp = m_vector;
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.to_dense(), temp);
		//~ noalias(x) += temp;
		//perform plus-assignment of temporary
		typename device_traits<device_type>:: template multiply_and_add<value_type> multiply(m_alpha);
		kernels::assign(x, temp, multiply);
	}
	matrix_closure_type m_matrix;
	vector_closure_type m_vector;
	value_type m_alpha;
};

template<class MatA, class F, class G>
class matrix_row_transform:public vector_expression<matrix_row_transform<MatA, F, G>, typename MatA::device_type >{
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
public:
	typedef typename G::result_type value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef matrix_row_transform const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename MatA::device_type device_type;
	typedef blockwise<typename MatA::evaluation_category::tag> evaluation_category;
	
	// Construction
	explicit matrix_row_transform(
		matrix_closure_type const& matrix, F const& f, G const& g
	):m_matrix(matrix), m_f(f), m_g(g){}

	// Accessors 
	size_type size() const{
		return m_matrix.size1();
	}
	matrix_closure_type const& matrix() const{
		return m_matrix;
	}
	F const& f() const{
		return m_f;
	}
	G const& g() const{
		return m_g;
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x)const{
		x().clear();
		plus_assign_to(x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x)const{
		kernels::fold_rows(eval_block(m_matrix), x, m_f, m_g);
	}
	
	// Iterator Access 
	typedef no_iterator const_iterator;
	typedef no_iterator iterator;
private:
	matrix_closure_type m_matrix;
	F m_f;
	G m_g;
};

//matrix-matrix prod
template<class MatA, class MatB>
class matrix_matrix_prod: public matrix_expression<matrix_matrix_prod<MatA, MatB>, typename MatA::device_type >{
public:
	typedef typename MatA::const_closure_type matrix_closure_typeA;
	typedef typename MatB::const_closure_type matrix_closure_typeB;
public:
	typedef decltype(
		typename MatA::value_type() * typename MatB::value_type()
	) value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_matrix_prod<MatA, MatB> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename evaluation_tag_restrict_traits<
		typename MatA::evaluation_category::tag,
		typename MatB::evaluation_category::tag
	>::type> evaluation_category;
	typedef unknown_orientation orientation;
	typedef typename MatA::device_type device_type;

	// Construction
	explicit matrix_matrix_prod(
		matrix_closure_typeA const& lhs,
		matrix_closure_typeB const& rhs,
		value_type alpha
	):m_lhs(lhs), m_rhs(rhs), m_alpha(alpha){}

	// Accessors 
	size_type size1() const{
		return m_lhs.size1();
	}
	size_type size2() const{
		return m_rhs.size2();
	}
	
	matrix_closure_typeA const& lhs() const{
		return m_lhs;
	}
	matrix_closure_typeB const& rhs() const{
		return m_rhs;
	}
	value_type alpha() const{
		return m_alpha;
	}
	
	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		assign_to(X, typename MatA::orientation(), typename MatA::storage_type::storage_tag());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		plus_assign_to(X, typename MatA::orientation(), typename MatA::storage_type::storage_tag());
	}
	
	// Iterator Access 
	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;
	
private:
	//gemm
	template<class MatX, class Category>
	void assign_to(matrix_expression<MatX, device_type>& X, linear_structure, Category c )const{
		X().clear();
		kernels::gemm(eval_block(m_lhs), eval_block(m_rhs), X, m_alpha);
	}
	template<class MatX, class Category>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, linear_structure, Category )const{
		kernels::gemm(eval_block(m_lhs), eval_block(m_rhs), X, m_alpha);
	}
	//trmv
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, triangular_structure, dense_tag)const{
		//assign the rhs and multiply in-place
		assign(X, m_rhs);
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.to_dense(), X);
		
		//perform multiplication with alpha if necessary
		if(m_alpha != value_type(1)){
			typedef typename device_traits<device_type>:: template multiply<value_type> Multiply;
			kernels::assign<Multiply>(X,m_alpha);
		}
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, triangular_structure, dense_tag )const{
		//computation of trmm is in-place so we need a temporary for plus-assign.
		typename matrix_temporary<MatX>::type temp = m_rhs;
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.to_dense(), temp);
		
		//perform plus-assignment of temporary
		typename device_traits<device_type>:: template multiply_and_add<value_type> multiply(m_alpha);
		kernels::assign(X, temp, multiply);
	}
private:
	matrix_closure_typeA m_lhs;
	matrix_closure_typeB m_rhs;
	value_type m_alpha;
};


/// \brief Concatenates two matrices A and B.
///
/// The third boolean argument decides whether this happens to the right as A|B 
/// or below as
/// A
/// --
/// B
template<class MatA, class MatB, bool add_right>
class matrix_concat: public matrix_expression<matrix_concat<MatA, MatB,  add_right>, typename MatA::device_type >{
public:
	typedef typename MatA::const_closure_type matrix_closure_typeA;
	typedef typename MatB::const_closure_type matrix_closure_typeB;
public:
	typedef typename common_value_type<MatA,MatB>::type value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const& const_reference;
	typedef const_reference reference;

	typedef matrix_concat const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef blockwise<typename evaluation_tag_restrict_traits<
		typename MatA::evaluation_category::tag,
		typename MatB::evaluation_category::tag
	>::type> evaluation_category;
	typedef unknown_orientation orientation;
	typedef typename MatA::device_type device_type;

	// Construction
	explicit matrix_concat(
		matrix_closure_typeA const& lhs,
		matrix_closure_typeB const& rhs
	):m_lhs(lhs), m_rhs(rhs){}

	// Accessors 
	size_type size1() const{
		return add_right? m_lhs.size1() : m_lhs.size1() + m_rhs.size1();
	}
	size_type size2() const{
		return add_right? m_lhs.size2() + m_rhs.size2() : m_lhs.size2();
	}
	
	matrix_closure_typeA const& lhs() const{
		return m_lhs;
	}
	matrix_closure_typeB const& rhs() const{
		return m_rhs;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//Element Functor
	no_functor elements() const{return no_functor();}
	
	// Computation Kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X)const{
		if(add_right){
			auto left = subrange(X,0, X().size1(), 0, m_lhs.size2()); 
			auto right = subrange(X,0, X().size1(), m_lhs.size2(), X().size2()); 
			assign(left,m_lhs);
			assign(right,m_rhs);
		}else{
			auto top = subrange(X,0, m_lhs.size1(), 0, X().size2()); 
			auto bottom = subrange(X,m_lhs.size1(), X().size1(),0, X().size2()); 
			assign(top,m_lhs);
			assign(bottom,m_rhs);
		}
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X)const{
		if(add_right){
			auto left = subrange(X,0, X().size1(), 0, m_lhs.size2()); 
			auto right = subrange(X,0, X().size1(), m_lhs.size2(), X().size2()); 
			plus_assign(left,m_lhs);
			plus_assign(right,m_rhs);
		}else{
			auto top = subrange(X,0, m_lhs.size1(), 0, X().size2()); 
			auto bottom = subrange(X,m_lhs.size1(), X().size1(),0, X().size2()); 
			plus_assign(top,m_lhs);
			plus_assign(bottom,m_rhs);
		}
	}
	
	// Iterator Access 
	typedef no_iterator const_major_iterator;
	typedef no_iterator major_iterator;
private:
	matrix_closure_typeA m_lhs;
	matrix_closure_typeB m_rhs;
};

}
#endif
