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

#include "../kernels/gemv.hpp"
#include "../kernels/tpmv.hpp"
#include "../kernels/trmv.hpp"
#include "../kernels/gemm.hpp"
#include "../kernels/trmm.hpp"
#include "../kernels/sum_rows.hpp"
#include "../assignment.hpp"
#include <type_traits>

namespace remora {

template<class E>
class matrix_scalar_multiply:public matrix_expression<matrix_scalar_multiply<E>, typename E::device_type > {
private:
	typedef typename device_traits<typename E::device_type>:: template multiply_scalar<typename E::value_type> functor_type;
public:
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

	// Construction and destruction
	explicit matrix_scalar_multiply(expression_closure_type const& e, value_type scalar):
		m_expression(e), m_scalar(scalar){}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	functor_type functor()const{
		return functor_type(m_scalar);
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

	// Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(this->functor()(this->expression()(i,j))){
		return functor()(m_expression(i,j));
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		m_expression.assign_to(X,alpha*m_scalar);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		m_expression.plus_assign_to(X,alpha*m_scalar);
	}

	// Iterator types
	typedef typename device_traits<device_type>:: template transform_iterator<
		typename E::const_row_iterator, functor_type
	> const_row_iterator;
	typedef typename device_traits<device_type>:: template transform_iterator<
		typename E::const_column_iterator, functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_expression.row_begin(i),functor_type(m_scalar));
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_expression.row_end(i),functor_type(m_scalar));
	}

	const_column_iterator column_begin(size_type i) const {
		return const_row_iterator(m_expression.column_begin(i),functor_type(m_scalar));
	}
	const_column_iterator column_end(size_type i) const {
		return const_row_iterator(m_expression.column_end(i),functor_type(m_scalar));
	}

private:
	expression_closure_type m_expression;
	value_type m_scalar;
};	
	
template<class E1, class E2>
class matrix_addition: public matrix_expression<matrix_addition<E1, E2>, typename E1::device_type > {
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
	typedef typename boost::mpl::if_< //the orientation is only known if E1 and E2 have the same
		std::is_same<typename E1::orientation, typename E2::orientation>,
		typename E1::orientation,
		unknown_orientation
	>::type orientation;
	//the evaluation category is blockwise if one of the expressions is blockwise or
	// if the orientation is unknown (efficient for expressions like A=B+C^T
	typedef typename boost::mpl::if_<
		std::is_same<orientation, unknown_orientation>,
		blockwise<typename evaluation_restrict_traits<E1,E2>::type::tag>,
		typename evaluation_restrict_traits<E1,E2>::type
	>::type evaluation_category;
	typedef typename E1::device_type device_type;

        // Construction
        explicit matrix_addition(
		lhs_closure_type const& e1,
		rhs_closure_type const& e2
	): m_lhs (e1), m_rhs (e2){}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1 ();
        }
        size_type size2 () const {
		return m_lhs.size2 ();
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

        // Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(functor_type()(this->lhs()(i,j),this->rhs()(i,j))){
		return functor_type()(lhs()(i,j),rhs()(i,j));
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		assign(X, m_lhs,alpha);
		plus_assign(X,m_rhs,alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		plus_assign(X,m_lhs,alpha);
		plus_assign(X,m_rhs,alpha);
	}

public:
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename E1::const_row_iterator,
		typename E2::const_row_iterator,
		functor_type
	> const_row_iterator;
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename E1::const_column_iterator,
		typename E2::const_column_iterator,
		functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(size_type j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(size_type j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_end(j),m_lhs.column_end(j),
			m_rhs.column_end(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

template<class V>
class vector_repeater:public matrix_expression<vector_repeater<V>, typename V::device_type > {
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
	typedef row_major orientation;
	typedef typename V::evaluation_category evaluation_category;

	// Construction and destruction
	explicit vector_repeater (expression_closure_type const& e, size_type rows):
	m_vector(e), m_rows(rows) {}

	// Accessors
	size_type size1() const {
		return m_rows;
	}
	size_type size2() const {
		return m_vector.size();
	}

	// Expression accessors
	const expression_closure_type& expression() const {
		return m_vector;
	}

	// Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& /*i*/, IndexExpr2 const& j) const -> decltype(this->expression()(j)){
		return m_vector(j);
	}
	
	typename device_traits<typename V::device_type>::queue_type& queue()const{
		return m_vector.queue();
	}

	// Iterator types
	typedef typename V::const_iterator const_row_iterator;
	typedef const_row_iterator row_iterator;
	typedef typename device_traits<typename V::device_type>:: template constant_iterator<value_type>  const_column_iterator;
	typedef const_column_iterator column_iterator;

	// Element lookup
	
	const_row_iterator row_begin(size_type i) const {
		RANGE_CHECK( i < size1());
		return m_vector.begin();
	}
	const_row_iterator row_end(size_type i) const {
		RANGE_CHECK( i < size1());
		return m_vector.end();
	}
	
	const_column_iterator column_begin(size_type j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(m_vector(j),0);
	}
	const_column_iterator column_end(size_type j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(m_vector(j),size1());
	}
private:
	expression_closure_type m_vector;
	size_type m_rows;
};

/// \brief A matrix with all values of type \c T equal to the same value
///
/// \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
template<class T, class Device>
class scalar_matrix:public matrix_container<scalar_matrix<T, Device>, Device > {
public:
	typedef std::size_t size_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;

	typedef scalar_matrix const_closure_type;
	typedef scalar_matrix closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef unknown_orientation orientation;
	typedef elementwise<dense_tag> evaluation_category;

	// Construction and destruction
	scalar_matrix():
		m_size1(0), m_size2(0), m_value() {}
	scalar_matrix(size_type size1, size_type size2, const value_type& value = value_type(1)):
		m_size1(size1), m_size2(size2), m_value(value) {}
	scalar_matrix(const scalar_matrix& m):
		m_size1(m.m_size1), m_size2(m.m_size2), m_value(m.m_value) {}

	// Accessors
	size_type size1() const {
		return m_size1;
	}
	size_type size2() const {
		return m_size2;
	}
	
	typename device_traits<Device>::queue_type& queue()const{
		return device_traits<Device>::default_queue();
	}

	// Element access
	template <class IndexExpr1, class IndexExpr2>
	value_type operator()(IndexExpr1 const&, IndexExpr2 const&) const{
		return m_value;
	}
	
	//Iterators
	typedef typename device_traits<Device>:: template constant_iterator<value_type> const_row_iterator;
	typedef typename device_traits<Device>:: template constant_iterator<value_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_value, 0);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_value, size2());
	}
	
	const_row_iterator column_begin(size_type j) const {
		return const_row_iterator(m_value, 0);
	}
	const_row_iterator column_end(size_type j) const {
		return const_row_iterator(m_value, size1());
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
class matrix_unary:public matrix_expression<matrix_unary<E, F>, typename E::device_type > {
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

	// Construction and destruction
	explicit matrix_unary(expression_closure_type const& e, functor_type const& functor):
		m_expression(e), m_functor(functor) {}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}
	
	expression_closure_type const& expression() const {
		return m_expression;
	}
	
	functor_type const& functor() const {
		return m_functor;
	}
	
	typename device_traits<device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha) const {
		X().clear();
		plus_assign_to(X,eval_block(m_expression), alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha) const {
		plus_assign_to(X,eval_block(m_expression), alpha);
	}

	// Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(this->functor()(this->expression()(i,j))){
		return functor()(m_expression(i,j));
	}

	// Iterator types
	typedef typename device_traits<device_type>:: template transform_iterator<typename E::const_row_iterator, F> const_row_iterator;
	typedef typename device_traits<device_type>:: template transform_iterator<typename E::const_column_iterator, F> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_expression.row_begin(i),m_functor);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_expression.row_end(i),m_functor);
	}

	const_column_iterator column_begin(size_type i) const {
		return const_column_iterator(m_expression.column_begin(i),m_functor);
	}
	const_column_iterator column_end(size_type i) const {
		return const_column_iterator(m_expression.column_end(i),m_functor);
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;

	template<class MatX, class MatA>
	void plus_assign_to(
		matrix_expression<MatX, device_type>& X,
		matrix_expression<MatA, device_type> const& A,
		value_type alpha
	)const{
		matrix_unary<MatA, F> e(A(), m_functor);
		plus_assign(X,e,alpha);
	}
};

template<class E1, class E2, class F>
class matrix_binary:public matrix_expression<matrix_binary<E1, E2, F>, typename E1::device_type > {
public:
	typedef E1 lhs_type;
	typedef E2 rhs_type;
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
	typedef typename boost::mpl::if_< //the orientation is only known if E1 and E2 have the same
		std::is_same<typename E1::orientation, typename E2::orientation>,
		typename E1::orientation,
		unknown_orientation
	>::type orientation;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;

	typedef F functor_type;

        // Construction and destruction

        explicit matrix_binary (
		lhs_closure_type const& e1,  rhs_closure_type const& e2, functor_type functor 
	): m_lhs (e1), m_rhs (e2),m_functor(functor) {}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1 ();
        }
        size_type size2 () const {
		return m_lhs.size2 ();
        }
	
	lhs_closure_type const& lhs() const {
		return m_lhs;
	}
	rhs_closure_type const& rhs() const {
		return m_rhs;
	}
	functor_type const& functor() const {
		return m_functor;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}

	// Element access
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(this->functor()(this->lhs()(i,j),this->rhs()(i,j))){
		return functor()(lhs()(i,j),rhs()(i,j));
	}
	
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha )const{
		X().clear();
		plus_assign_to(X,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		plus_assign_to(X,eval_block(m_lhs), eval_block(m_rhs), alpha);
	}

	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename E1::const_row_iterator,typename E2::const_row_iterator,functor_type
	> const_row_iterator;
	typedef typename device_traits<device_type>:: template binary_transform_iterator<
		typename E1::const_column_iterator,typename E2::const_column_iterator,functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator (m_functor,
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator (m_functor,
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(size_type j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(size_type j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_end(j),m_lhs.column_end(j),
			m_rhs.column_end(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;

	template<class MatX, class LHS, class RHS>
	void plus_assign_to(
		matrix_expression<MatX, device_type>& X,
		matrix_expression<LHS, device_type> const& lhs,
		matrix_expression<RHS, device_type> const& rhs,
		value_type alpha
	)const{
		//we know that lhs and rhs are elementwise expressions so we can now create the elementwise expression and assign it.
		matrix_binary<LHS,RHS,F> e(lhs(),rhs(), m_functor);
		plus_assign(X,e,alpha);
	}
};

template<class E1, class E2>
class outer_product:public matrix_expression<outer_product<E1, E2>, typename E1::device_type > {
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
	typedef unknown_orientation orientation;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;
	typedef typename E1::device_type device_type;
	// Construction and destruction
	
	explicit outer_product(lhs_closure_type const& e1, rhs_closure_type const& e2)
	:m_lhs(e1), m_rhs(e2){}

	// Accessors
	size_type size1() const {
		return m_lhs.size();
	}
	size_type size2() const {
		return m_rhs.size();
	}

	// Expression accessors
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
	template <class IndexExpr1, class IndexExpr2>
	auto operator()(IndexExpr1 const& i, IndexExpr2 const& j) const -> decltype(functor_type_op()(this->lhs()(i),this->rhs()(j))){
		return functor_type_op()(lhs()(i),rhs()(j));
	}

	typedef typename device_traits<device_type>:: template transform_iterator<typename E2::const_iterator,functor_type> const_row_iterator;
	typedef typename device_traits<device_type>:: template transform_iterator<typename E1::const_iterator,functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(size_type i) const {
		return const_row_iterator(m_rhs.begin(),
			functor_type(m_lhs(i))
		);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(m_rhs.end(),
			functor_type(m_lhs(i))
		);
	}

	const_column_iterator column_begin(size_type i) const {
		return const_column_iterator(m_lhs.begin(),
			functor_type(m_rhs(i))
		);
	}
	const_column_iterator column_end(size_type i) const {
		return const_column_iterator(m_lhs.end(),
			functor_type(m_rhs(i))
		);
	}
private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};

template<class MatA, class VecV>
class matrix_vector_prod:public vector_expression<matrix_vector_prod<MatA, VecV>, typename MatA::device_type > {
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

	// Construction and destruction
	explicit matrix_vector_prod(
		matrix_closure_type const& matrix,
		vector_closure_type const& vector
	):m_matrix(matrix), m_vector(vector) {}

	size_type size() const {
		return m_matrix.size1();
	}
	
	matrix_closure_type const& matrix() const {
		return m_matrix;
	}
	vector_closure_type const& vector() const {
		return m_vector;
	}
	
	typedef typename MatA::const_row_iterator const_iterator;
	typedef const_iterator iterator;


	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	//dispatcher to computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, value_type alpha )const{
		assign_to(x, alpha, typename MatA::orientation(), typename MatA::evaluation_category::tag());
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, value_type alpha)const{
		plus_assign_to(x, alpha, typename MatA::orientation(), typename MatA::evaluation_category::tag());
	}
	
private:
	//gemv
	template<class VecX, class Category>
	void assign_to(vector_expression<VecX, device_type>& x, value_type alpha, linear_structure, Category c)const{
		x().clear();
		plus_assign_to(x,alpha, linear_structure(), c);
	}
	template<class VecX, class Category>
	void plus_assign_to(vector_expression<VecX, device_type>& x, value_type alpha, linear_structure, Category )const{
		kernels::gemv(eval_block(m_matrix), eval_block(m_vector), x, alpha);
	}
	//tpmv
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, value_type alpha, triangular_structure, packed_tag )const{
		noalias(x) = eval_block(alpha * m_vector);
		kernels::tpmv(eval_block(m_matrix), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, value_type alpha, triangular_structure, packed_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp( eval_block(alpha * m_vector));
		kernels::tpmv(eval_block(m_matrix), temp);
		noalias(x) += temp;
	}
	//trmv
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, value_type alpha, triangular_structure, dense_tag )const{
		noalias(x) = eval_block(alpha * m_vector);
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.expression(), x);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, value_type alpha, triangular_structure, dense_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename vector_temporary<VecX>::type temp( eval_block(alpha * m_vector));
		kernels::trmv<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_matrix.expression(), temp);
		noalias(x) += temp;
	}
	matrix_closure_type m_matrix;
	vector_closure_type m_vector;
};


template<class MatA>
class sum_matrix_rows:public vector_expression<sum_matrix_rows<MatA>, typename MatA::device_type > {
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
public:
	typedef typename MatA::value_type value_type;
	typedef typename MatA::size_type size_type;
	typedef value_type const_reference;
	typedef const_reference reference;

	typedef sum_matrix_rows const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef typename MatA::device_type device_type;
	typedef blockwise<typename MatA::evaluation_category::tag> evaluation_category;
	

	explicit sum_matrix_rows(
		matrix_closure_type const& matrix
	):m_matrix(matrix){}

	size_type size() const {
		return m_matrix.size2();
	}
	
	matrix_closure_type const& matrix() const {
		return m_matrix;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_matrix.queue();
	}
	
	typedef typename MatA::const_row_iterator const_iterator;
	typedef const_iterator iterator;

	//dispatcher to computation kernels for blockwise case
	template<class VecX>
	void assign_to(vector_expression<VecX, device_type>& x, value_type alpha)const{
		x().clear();
		plus_assign_to(x,alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX, device_type>& x, value_type alpha)const{
		kernels::sum_rows(eval_block(m_matrix), x, alpha);
	}
private:
	matrix_closure_type m_matrix;
};

namespace detail{
template<class M, class TriangularType>
class dense_triangular_proxy: public matrix_expression<dense_triangular_proxy<M, TriangularType>, typename M::device_type > {
	typedef typename closure<M>::type matrix_closure_type;
public:
	static_assert(std::is_same<typename M::storage_type::storage_tag, dense_tag>::value, "Can only create triangular proxies of dense matrices");

	typedef typename M::size_type size_type;
	typedef typename M::value_type value_type;
	typedef typename M::const_reference const_reference;
	typedef typename reference<M>::type reference;
	typedef dense_triangular_proxy<typename const_expression<M>::type,TriangularType> const_closure_type;
	typedef dense_triangular_proxy<M,TriangularType> closure_type;

	typedef dense_matrix_storage<value_type> storage_type;
	typedef dense_matrix_storage<value_type const> const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef triangular<typename M::orientation,TriangularType> orientation;

	// Construction and destruction
	explicit dense_triangular_proxy(matrix_closure_type const& m)
	: m_expression(m) {
		SIZE_CHECK(m.size1() == m.size2());
	}

	// Expression accessors
	matrix_closure_type const& expression() const{
		return m_expression;
	}
	matrix_closure_type expression(){
		return m_expression;
	}
	
	 size_type size1 () const {
		return expression().size1 ();
        }
        size_type size2 () const {
		return expression().size2 ();
        }
	
	typename device_traits<typename M::device_type>::queue_type& queue()const{
		return m_expression.queue();
	}
	
	///\brief Returns the underlying storage_type structure for low level access
	storage_type raw_storage_type(){
		return expression().raw_storage_type();
	}
	
	typedef typename M::const_row_iterator row_iterator;
	typedef row_iterator const_row_iterator;
	typedef typename M::const_column_iterator column_iterator;
	typedef column_iterator const_column_iterator;
private:
	matrix_closure_type m_expression;
};
}

//matrix-matrix prod
template<class MatA, class MatB>
class matrix_matrix_prod: public matrix_expression<matrix_matrix_prod<MatA, MatB>, typename MatA::device_type > {
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

	// Construction and destruction
	explicit matrix_matrix_prod(
		matrix_closure_typeA const& lhs,
		matrix_closure_typeB const& rhs
	):m_lhs(lhs), m_rhs(rhs) {}

	size_type size1() const {
		return m_lhs.size1();
	}
	size_type size2() const {
		return m_rhs.size2();
	}
	
	matrix_closure_typeA const& lhs() const {
		return m_lhs;
	}
	matrix_closure_typeB const& rhs() const {
		return m_rhs;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//FIXME: This workaround is required
	//as other expressions might query the iterator type
	typedef typename MatA::const_row_iterator const_row_iterator;
	typedef typename MatA::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		assign_to(X, alpha, typename MatA::orientation(), typename MatA::storage_type::storage_tag());
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		plus_assign_to(X, alpha, typename MatA::orientation(), typename MatA::storage_type::storage_tag());
	}
	
private:
	//gemm
	template<class MatX, class Category>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha, linear_structure, Category c )const{
		X().clear();
		plus_assign_to(X,alpha, linear_structure(), c);
	}
	template<class MatX, class Category>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha, linear_structure, Category )const{
		kernels::gemm(eval_block(m_lhs), eval_block(m_rhs), X, alpha);
	}
	//trmv
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha, triangular_structure, dense_tag)const{
		noalias(X) = eval_block(alpha * m_rhs);
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.expression(), X);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha, triangular_structure, dense_tag )const{
		//computation of tpmv is in-place so we need a temporary for plus-assign.
		typename matrix_temporary<MatX>::type temp( eval_block(alpha * m_rhs));
		kernels::trmm<MatA::orientation::is_upper, MatA::orientation::is_unit>(m_lhs.expression(), temp);
		noalias(X) += temp;
	}
private:
	matrix_closure_typeA m_lhs;
	matrix_closure_typeB m_rhs;
};

/// \brief An diagonal matrix with values stored inside a diagonal vector
///
/// the matrix stores a Vector representing the diagonal.
template<class V>
class diagonal_matrix: public matrix_expression<diagonal_matrix< V >, typename V::device_type > {
	typedef typename V::const_closure_type vector_closure_type;
public:
	typedef typename V::size_type size_type;
	typedef typename V::value_type value_type;
	typedef value_type  const_reference;
	typedef value_type reference;

	typedef diagonal_matrix const_closure_type;
	typedef diagonal_matrix closure_type;
	typedef unknown_storage storage_type;
	typedef unknown_storage const_storage_type;
	typedef elementwise<dense_tag> evaluation_category;
	typedef unknown_orientation orientation;
	typedef typename V::device_type device_type;

	// Construction and destruction
	diagonal_matrix(vector_closure_type const& diagonal):m_diagonal(diagonal){}

	// Accessors
	size_type size1() const {
		return m_diagonal.size();
	}
	size_type size2() const {
		return m_diagonal.size();
	}
	typename device_traits<device_type>::queue_type& queue()const{
		return m_diagonal.queue();
	}
	
	vector_closure_type const& expression()const{
		return m_diagonal;
	}
	// Element access
	const_reference operator()(size_type i, size_type j) const {
		if (i == j)
			return m_diagonal(i);
		else
			return value_type();
	}
	
	//Iterators
	typedef typename device_traits<device_type>:: template one_hot_iterator<value_type const> const_row_iterator;
	typedef const_row_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	
	const_row_iterator row_begin(size_type i) const {
		return row_iterator(i, m_diagonal(i),false);
	}
	const_row_iterator row_end(size_type i) const {
		return const_row_iterator(i, value_type(),true);
	}
	const_column_iterator column_begin(size_type i) const {
		return column_iterator(i, m_diagonal(i),false);
	}
	const_column_iterator column_end(size_type i) const {
		return const_column_iterator(i, value_type(),true);
	}

private:
	vector_closure_type m_diagonal; 
};


/// \brief Concatenates two matrices A and B.
///
/// The third boolean argument decides whether this happens to the right as A|B 
/// or below as
/// A
/// --
/// B
template<class MatA, class MatB, bool add_right>
class matrix_concat: public matrix_expression<matrix_concat<MatA, MatB,  add_right>, typename MatA::device_type > {
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

	// Construction and destruction
	explicit matrix_concat(
		matrix_closure_typeA const& lhs,
		matrix_closure_typeB const& rhs
	):m_lhs(lhs), m_rhs(rhs) {}

	size_type size1() const {
		return add_right? m_lhs.size1() : m_lhs.size1() + m_rhs.size1();
	}
	size_type size2() const {
		return add_right? m_lhs.size2() + m_rhs.size2() : m_lhs.size2();
	}
	
	matrix_closure_typeA const& lhs() const {
		return m_lhs;
	}
	matrix_closure_typeB const& rhs() const {
		return m_rhs;
	}

	typename device_traits<device_type>::queue_type& queue()const{
		return m_lhs.queue();
	}
	
	//FIXME: This workaround is required
	//as other expressions might query the iterator type
	typedef typename MatA::const_row_iterator const_row_iterator;
	typedef typename MatA::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	//dispatcher to computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		if(add_right){
			matrix_range<MatX> left(X(),0, X().size1(), 0, m_lhs.size2()); 
			matrix_range<MatX> right(X(),0, X().size1(), m_lhs.size2(), X().size2()); 
			assign(left,m_lhs,alpha);
			assign(right,m_rhs,alpha);
		}else{
			matrix_range<MatX> top(X(),0, m_lhs.size1(), 0, X().size2()); 
			matrix_range<MatX> bottom(X(),m_lhs.size1(), X().size1(),0, X().size2()); 
			assign(top,m_lhs,alpha);
			assign(bottom,m_rhs,alpha);
		}
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX, device_type>& X, value_type alpha)const{
		if(add_right){
			matrix_range<MatX> left(X(),0, X().size1(), 0, m_lhs.size2()); 
			matrix_range<MatX> right(X(),0, X().size1(), m_lhs.size2(), X().size2()); 
			plus_assign(left,m_lhs,alpha);
			plus_assign(right,m_rhs,alpha);
		}else{
			matrix_range<MatX> top(X(),0, m_lhs.size1(), 0, X().size2()); 
			matrix_range<MatX> bottom(X(),m_lhs.size1(), X().size1(),0, X().size2()); 
			plus_assign(top,m_lhs,alpha);
			plus_assign(bottom,m_rhs,alpha);
		}
	}
private:
	matrix_closure_typeA m_lhs;
	matrix_closure_typeB m_rhs;
};

}
#endif
