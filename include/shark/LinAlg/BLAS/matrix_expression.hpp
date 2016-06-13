/*!
 * \brief       Expression templates for expressions involving matrices
 * 
 * \author      O. Krause
 * \date        2013
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
#ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP

#include "matrix_proxy.hpp"
#include "operation.hpp"

#include <boost/utility/enable_if.hpp>
#include <type_traits>

namespace shark {
namespace blas {

template<class E1, class E2>
class outer_product:public matrix_expression<outer_product<E1, E2> > {
	typedef scalar_multiply1<typename E1::value_type, typename E2::value_type> functor_type;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef outer_product const_closure_type;
	typedef outer_product closure_type;
	typedef unknown_orientation orientation;
	typedef unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	// Construction and destruction
	
	outer_product(lhs_closure_type const& e1, rhs_closure_type const& e2)
	:m_lhs(e1), m_rhs(e2){}

	// Accessors
	size_type size1() const {
		return m_lhs.size();
	}
	size_type size2() const {
		return m_rhs.size();
	}

	// Expression accessors
	const lhs_closure_type &lhs() const {
		return m_lhs;
	}
	const rhs_closure_type &rhs() const {
		return m_rhs;
	}
	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_lhs(i) * m_rhs(j);
	}

	typedef transform_iterator<typename E2::const_iterator,functor_type> const_row_iterator;
	typedef transform_iterator<typename E1::const_iterator,functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_rhs.begin(),
			functor_type(m_lhs(i))
		);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_rhs.end(),
			functor_type(m_lhs(i))
		);
	}

	const_column_iterator column_begin(index_type i) const {
		return const_column_iterator(m_lhs.begin(),
			functor_type(m_rhs(i))
		);
	}
	const_column_iterator column_end(index_type i) const {
		return const_column_iterator(m_lhs.end(),
			functor_type(m_rhs(i))
		);
	}
private:
	lhs_closure_type m_lhs;
	rhs_closure_type m_rhs;
};

// (outer_prod (v1, v2)) [i] [j] = v1 [i] * v2 [j]
template<class E1, class E2>
outer_product<E1, E2 >
outer_prod(
	vector_expression<E1> const& e1,
        vector_expression<E2> const& e2
) {
	return outer_product<E1, E2>(e1(), e2());
}

template<class V>
class vector_repeater:public blas::matrix_expression<vector_repeater<V> > {
private:
	typedef V expression_type;
	typedef vector_repeater<V> self_type;
	typedef typename V::const_iterator const_subiterator_type;
public:
	typedef typename V::const_closure_type expression_closure_type;
	typedef typename V::size_type size_type;
	typedef typename V::difference_type difference_type;
	typedef typename V::value_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* pointer;
	typedef value_type const* const_pointer;

	typedef typename V::index_type index_type;
	typedef typename V::const_index_pointer const_index_pointer;
	typedef typename index_pointer<V>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef blas::row_major orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename V::evaluation_category evaluation_category;

	// Construction and destruction
	explicit vector_repeater (expression_type const& e, std::size_t rows):
	m_vector(e), m_rows(rows) {}

	// Accessors
	size_type size1() const {
		return m_rows;
	}
	size_type size2() const {
		return m_vector.size();
	}

	// Expression accessors
	const expression_closure_type &expression () const {
		return m_vector;
	}

	// Element access
	const_reference operator() (index_type i, index_type j) const {
		return m_vector(j);
	}

	// Iterator types
	typedef typename V::const_iterator const_row_iterator;
	typedef const_row_iterator row_iterator;
	typedef blas::constant_iterator<value_type>  const_column_iterator;
	typedef const_column_iterator column_iterator;

	// Element lookup
	
	const_row_iterator row_begin(std::size_t i) const {
		RANGE_CHECK( i < size1());
		return m_vector.begin();
	}
	const_row_iterator row_end(std::size_t i) const {
		RANGE_CHECK( i < size1());
		return m_vector.end();
	}
	
	const_column_iterator column_begin(std::size_t j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(0,m_vector(j));
	}
	const_column_iterator column_end(std::size_t j) const {
		RANGE_CHECK( j < size2());
		return const_column_iterator(size1(),m_vector(j));
	}
private:
	expression_closure_type m_vector;
	std::size_t m_rows;
};

///\brief Creates a matrix from a vector by repeating the vector in every row of the matrix
///
///example: vector = (1,2,3)
///repeat(vector,3) results in
///(1,2,3)
///(1,2,3)
///(1,2,3)
///@param vector the vector which is to be repeated as the rows of the resulting matrix
///@param rows the number of rows of the matrix
template<class Vector>
vector_repeater<Vector> repeat(vector_expression<Vector> const& vector, std::size_t rows){
	return vector_repeater<Vector>(vector(),rows);
}

/** \brief A matrix with all values of type \c T equal to the same value
 *
 * \tparam T the type of object stored in the matrix (like double, float, complex, etc...)
 */
template<class T>
class scalar_matrix:
	public matrix_container<scalar_matrix<T> > {

	typedef scalar_matrix<T> self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef T value_type;
	typedef const T& const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef value_type scalar_type;
	typedef const_pointer pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef self_type const_closure_type;
	typedef self_type closure_type;
	typedef dense_tag storage_category;
	typedef unknown_orientation orientation;
	typedef elementwise_tag evaluation_category;

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

	// Element access
	const_reference operator()(size_type /*i*/, size_type /*j*/) const {
		return m_value;
	}
	
	//Iterators
	typedef constant_iterator<value_type> const_row_iterator;
	typedef constant_iterator<value_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator(size2(), m_value);
	}
	
	const_row_iterator column_begin(std::size_t j) const {
		return const_row_iterator(0, m_value);
	}
	const_row_iterator column_end(std::size_t j) const {
		return const_row_iterator(size1(), m_value);
	}
private:
	size_type m_size1;
	size_type m_size2;
	value_type m_value;
};

///brief repeats a single element to form a matrix  of size rows x columns
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_matrix<T> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T>(rows, columns, scalar);
}

template<class E>
class matrix_scalar_multiply:public blas::matrix_expression<matrix_scalar_multiply<E> > {
private:
	typedef typename E::const_row_iterator const_subrow_iterator_type;
	typedef typename E::const_column_iterator const_subcolumn_iterator_type;
	typedef scalar_multiply1<typename E::value_type, typename E::scalar_type> functor_type;
public:
	typedef typename E::const_closure_type expression_closure_type;

	typedef typename functor_type::result_type value_type;
	typedef typename E::scalar_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const *const_pointer;
	typedef value_type *pointer;
	typedef typename E::size_type size_type;
	typedef typename E::difference_type difference_type;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef matrix_scalar_multiply const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	matrix_scalar_multiply(blas::matrix_expression<E> const &e, scalar_type scalar):
		m_expression(e()), m_scalar(scalar){}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_scalar * m_expression(i, j);
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.assign_to(X,alpha*m_scalar);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.plus_assign_to(X,alpha*m_scalar);
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		m_expression.minus_assign_to(X,alpha*m_scalar);
	}

	// Iterator types
	typedef transform_iterator<typename E::const_row_iterator, functor_type> const_row_iterator;
	typedef transform_iterator<typename E::const_column_iterator, functor_type> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression.row_begin(i),functor_type(m_scalar));
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression.row_end(i),functor_type(m_scalar));
	}

	const_column_iterator column_begin(index_type i) const {
		return const_row_iterator(m_expression.column_begin(i),functor_type(m_scalar));
	}
	const_column_iterator column_end(index_type i) const {
		return const_row_iterator(m_expression.column_end(i),functor_type(m_scalar));
	}

private:
	expression_closure_type m_expression;
	scalar_type m_scalar;
};


///\brief class which allows for matrix transformations
///
///transforms a matrix expression e of type E using a Functiof f of type F as an elementwise transformation f(e(i,j))
///This transformation needs to leave f constant, meaning that applying f(x), f(y), f(z) yields the same results independent of the
///order of application. Also F must provide a type F::result_type indicating the result type of the functor.
///F must further provide a boolean flag F::zero_identity which indicates that f(0) = 0. This is needed for correct usage with sparse
///arguments - if f(0) != 0 this expression will be dense!
///todo: desification is not implemented
template<class E, class F>
class matrix_unary:public blas::matrix_expression<matrix_unary<E, F> > {
private:
	typedef matrix_unary<E, F> self_type;
	typedef E expression_type;
	typedef typename expression_type::const_row_iterator const_subrow_iterator_type;
	typedef typename expression_type::const_column_iterator const_subcolumn_iterator_type;

public:
	typedef typename expression_type::const_closure_type expression_closure_type;

	typedef F functor_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const *const_pointer;
	typedef value_type *pointer;
	typedef typename expression_type::size_type size_type;
	typedef typename expression_type::difference_type difference_type;

	typedef typename E::index_type index_type;
	typedef typename E::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E>::type index_pointer;

	typedef self_type const const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename E::evaluation_category evaluation_category;

	// Construction and destruction
	matrix_unary(blas::matrix_expression<E> const &e, F const &functor):
		m_expression(e()), m_functor(functor) {}

	// Accessors
	size_type size1() const {
		return m_expression.size1();
	}
	size_type size2() const {
		return m_expression.size2();
	}

	// Element access
	const_reference operator()(index_type i, index_type j) const {
		return m_functor(m_expression(i, j));
	}

	// Iterator types
	typedef transform_iterator<typename E::const_row_iterator, F> const_row_iterator;
	typedef transform_iterator<typename E::const_column_iterator, F> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;
	
	const_row_iterator row_begin(index_type i) const {
		return const_row_iterator(m_expression.row_begin(i),m_functor);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(m_expression.row_end(i),m_functor);
	}

	const_column_iterator column_begin(index_type i) const {
		return const_row_iterator(m_expression.column_begin(i),m_functor);
	}
	const_column_iterator column_end(index_type i) const {
		return const_row_iterator(m_expression.column_end(i),m_functor);
	}

private:
	expression_closure_type m_expression;
	functor_type m_functor;
};

template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (matrix_expression<E> const& e, T scalar){
	return matrix_scalar_multiply<E>(e(), typename E::scalar_type(scalar));
}

template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (T scalar, matrix_expression<E> const& e){
	return matrix_scalar_multiply<E>(e(), typename E::scalar_type(scalar));
}

template<class E>
matrix_scalar_multiply<E> operator-(matrix_expression<E> const& e){
	return matrix_scalar_multiply<E>(e(), typename E::scalar_type(-1));
}

#define SHARK_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class E>\
matrix_unary<E,F<typename E::value_type> >\
name(matrix_expression<E> const& e){\
	typedef F<typename E::value_type> functor_type;\
	return matrix_unary<E, functor_type>(e, functor_type());\
}
SHARK_UNARY_MATRIX_TRANSFORMATION(conj, scalar_conj)
SHARK_UNARY_MATRIX_TRANSFORMATION(real, scalar_real)
SHARK_UNARY_MATRIX_TRANSFORMATION(imag, scalar_imag)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_MATRIX_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_MATRIX_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_MATRIX_TRANSFORMATION(sin, scalar_sin)
SHARK_UNARY_MATRIX_TRANSFORMATION(cos, scalar_cos)
SHARK_UNARY_MATRIX_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(atanh,scalar_atanh)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_MATRIX_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_MATRIX_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_MATRIX_TRANSFORMATION(softPlus, scalar_soft_plus)
SHARK_UNARY_MATRIX_TRANSFORMATION(elem_inv, scalar_inverse)
#undef SHARK_UNARY_MATRIX_TRANSFORMATION

#define SHARK_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class E, class T> \
typename boost::enable_if< \
	std::is_convertible<T, typename E::value_type >,\
        matrix_unary<E,F<typename E::value_type,T> > \
>::type \
name (matrix_expression<E> const& e, T scalar){ \
	typedef F<typename E::value_type, T> functor_type; \
	return matrix_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
SHARK_MATRIX_SCALAR_TRANSFORMATION(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION(max, scalar_max)
SHARK_MATRIX_SCALAR_TRANSFORMATION(pow, scalar_pow)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define SHARK_MATRIX_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	std::is_convertible<T, typename E::value_type >,\
        matrix_unary<E,F<typename E::value_type,T> > \
>::type \
name (T scalar, matrix_expression<E> const& e){ \
	typedef F<typename E::value_type, T> functor_type; \
	return matrix_unary<E, functor_type>(e, functor_type(scalar)); \
}
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(min, scalar_min)
SHARK_MATRIX_SCALAR_TRANSFORMATION_2(max, scalar_max)
#undef SHARK_MATRIX_SCALAR_TRANSFORMATION_2

template<class E1, class E2>
class matrix_addition: public blas::matrix_expression<matrix_addition<E1, E2> > {
private:
	typedef scalar_binary_plus<
		typename E1::value_type,
		typename E2::value_type
	> functor_type;
public:
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename functor_type::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef const matrix_addition<E1, E2> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

        // Construction
        matrix_addition(
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

        const_reference operator () (index_type i, index_type j) const {
		return m_lhs(i, j) + m_rhs(i,j);
        }
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		assign(X,alpha * m_lhs);
		plus_assign(X,alpha * m_rhs);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		plus_assign(X,alpha * m_lhs);
		plus_assign(X,alpha * m_rhs);
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		minus_assign(X,alpha * m_lhs);
		minus_assign(X,alpha * m_rhs);
	}

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_row_iterator1_type;
	typedef typename E1::const_column_iterator const_row_column_iterator_type;
	typedef typename E2::const_row_iterator const_column_iterator1_type;
	typedef typename E2::const_column_iterator const_column_iterator2_type;

public:
	typedef binary_transform_iterator<
		typename E1::const_row_iterator,
		typename E2::const_row_iterator,
		functor_type
	> const_row_iterator;
	typedef binary_transform_iterator<
		typename E1::const_column_iterator,
		typename E2::const_column_iterator,
		functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (functor_type(),
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (functor_type(),
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

///\brief Adds two Matrices
template<class E1, class E2>
matrix_addition<E1, E2 > operator+ (
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2
){
	return matrix_addition<E1, E2>(e1(),e2());
}

///\brief Subtracts two Matrices
template<class E1, class E2>
matrix_addition<E1, matrix_scalar_multiply<E2> > operator- (
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2
){
	return matrix_addition<E1, matrix_scalar_multiply<E2> >(e1(),-e2());
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>, 
	matrix_addition<E, scalar_matrix<T> >
>::type operator+ (
	matrix_expression<E> const& e,
	T t
){
	return e + scalar_matrix<T>(e().size1(),e().size2(),t);
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>,
	matrix_addition<E, scalar_matrix<T> >
>::type operator+ (
	T t,
	matrix_expression<E> const& e
){
	return e + scalar_matrix<T>(e().size1(),e().size2(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant matrix from a matrix.
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type> ,
	matrix_addition<E, matrix_scalar_multiply<scalar_matrix<T> > >
>::type operator- (
	matrix_expression<E> const& e,
	T t
){
	return e - scalar_matrix<T>(e().size1(),e().size2(),t);
}

///\brief Subtracts a matrix from a scalar which is interpreted as a constant matrix
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>,
	matrix_addition<scalar_matrix<T>, matrix_scalar_multiply<E> >
>::type operator- (
	T t,
	matrix_expression<E> const& e
){
	return scalar_matrix<T>(e().size1(),e().size2(),t) - e;
}

template<class E1, class E2, class F>
class matrix_binary:
	public blas::matrix_expression<matrix_binary<E1, E2, F> > {
private:
	typedef matrix_binary<E1, E2, F> self_type;
public:
	typedef E1 lhs_type;
	typedef E2 rhs_type;
	typedef typename E1::const_closure_type lhs_closure_type;
	typedef typename E2::const_closure_type rhs_closure_type;

	typedef typename E1::size_type size_type;
	typedef typename E1::difference_type difference_type;
	typedef typename F::result_type value_type;
	typedef value_type scalar_type;
	typedef value_type const_reference;
	typedef const_reference reference;
	typedef value_type const* const_pointer;
	typedef const_pointer pointer;

	typedef typename E1::index_type index_type;
	typedef typename E1::const_index_pointer const_index_pointer;
	typedef typename index_pointer<E1>::type index_pointer;

	typedef const matrix_binary<E1, E2, F> const_closure_type;
	typedef const_closure_type closure_type;
	typedef typename E1::orientation orientation;
	typedef blas::unknown_storage_tag storage_category;
	typedef typename evaluation_restrict_traits<E1,E2>::type evaluation_category;

	typedef F functor_type;

        // Construction and destruction

        matrix_binary (
		matrix_expression<E1> const&e1,  matrix_expression<E2> const& e2, functor_type functor 
	): m_lhs (e1()), m_rhs (e2()),m_functor(functor) {}

        // Accessors
        size_type size1 () const {
		return m_lhs.size1 ();
        }
        size_type size2 () const {
		return m_lhs.size2 ();
        }

        const_reference operator () (index_type i, index_type j) const {
		return m_functor( m_lhs (i, j), m_rhs(i,j));
        }

	// Iterator types
private:
	typedef typename E1::const_row_iterator const_row_iterator1_type;
	typedef typename E1::const_column_iterator const_row_column_iterator_type;
	typedef typename E2::const_row_iterator const_column_iterator1_type;
	typedef typename E2::const_column_iterator const_column_iterator2_type;

public:
	typedef binary_transform_iterator<
		typename E1::const_row_iterator,typename E2::const_row_iterator,functor_type
	> const_row_iterator;
	typedef binary_transform_iterator<
		typename E1::const_column_iterator,typename E2::const_column_iterator,functor_type
	> const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	const_row_iterator row_begin(std::size_t i) const {
		return const_row_iterator (m_functor,
			m_lhs.row_begin(i),m_lhs.row_end(i),
			m_rhs.row_begin(i),m_rhs.row_end(i)
		);
	}
	const_row_iterator row_end(std::size_t i) const {
		return const_row_iterator (m_functor,
			m_lhs.row_end(i),m_lhs.row_end(i),
			m_rhs.row_end(i),m_rhs.row_end(i)
		);
	}

	const_column_iterator column_begin(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}
	const_column_iterator column_end(std::size_t j) const {
		return const_column_iterator (m_functor,
			m_lhs.column_begin(j),m_lhs.column_end(j),
			m_rhs.column_begin(j),m_rhs.column_end(j)
		);
	}

private:
	lhs_closure_type m_lhs;
        rhs_closure_type m_rhs;
	functor_type m_functor;
};

#define SHARK_BINARY_MATRIX_EXPRESSION(name, F)\
template<class E1, class E2>\
matrix_binary<E1, E2, F<typename E1::value_type, typename E2::value_type> >\
name(matrix_expression<E1> const& e1, matrix_expression<E2> const& e2){\
	typedef F<typename E1::value_type, typename E2::value_type> functor_type;\
	return matrix_binary<E1, E2, functor_type>(e1,e2, functor_type());\
}
SHARK_BINARY_MATRIX_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_MATRIX_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_MATRIX_EXPRESSION(pow,scalar_binary_pow)
SHARK_BINARY_MATRIX_EXPRESSION(element_div, scalar_binary_divide)
#undef SHARK_BINARY_MATRIX_EXPRESSION

template<class E1, class E2>
matrix_binary<E1, E2, 
	scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> 
>
safe_div(
	matrix_expression<E1> const& e1, 
	matrix_expression<E2> const& e2, 
	decltype(typename E1::value_type()/typename E2::value_type()) defaultValue
){
	typedef scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> functor_type;
	return matrix_binary<E1, E2, functor_type>(e1,e2, functor_type(defaultValue));
}

template<class MatA, class VecV>
class matrix_vector_prod:
	public vector_expression<matrix_vector_prod<MatA, VecV> > {
public:
	typedef typename MatA::const_closure_type matrix_closure_type;
	typedef typename VecV::const_closure_type vector_closure_type;
public:
	typedef decltype(
		typename MatA::scalar_type() * typename VecV::scalar_type()
	) scalar_type;
	typedef scalar_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;
	typedef typename MatA::index_pointer index_pointer;
	typedef typename MatA::const_index_pointer const_index_pointer;

	typedef matrix_vector_prod<MatA, VecV> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;


	//FIXME: This workaround is required to be able to generate
	// temporary vectors
	typedef typename MatA::const_row_iterator const_iterator;
	typedef const_iterator iterator;

	// Construction and destruction
	matrix_vector_prod(
		matrix_closure_type const& matrix,
		vector_closure_type  const& vector
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
	
	//computation kernels
	template<class VecX>
	void assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		x().clear();
		plus_assign_to(x,alpha);
	}
	template<class VecX>
	void plus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		kernels::gemv(eval_block(m_matrix), eval_block(m_vector), x, alpha);
	}
	
	template<class VecX>
	void minus_assign_to(vector_expression<VecX>& x, scalar_type alpha = scalar_type(1) )const{
		kernels::gemv(eval_block(m_matrix), eval_block(m_vector), x, -alpha);
	}
	
private:
	matrix_closure_type m_matrix;
	vector_closure_type m_vector;
};


/// \brief computes the matrix-vector product x+=Av
template<class MatA, class VecV>
matrix_vector_prod<MatA,VecV> prod(matrix_expression<MatA> const& A,vector_expression<VecV> const& v) {
	return matrix_vector_prod<MatA,VecV>(A(),v());
}

/// \brief computes the matrix-vector product x+=v^TA
template<class MatA, class VecV>
matrix_vector_prod<matrix_transpose<MatA>,VecV> prod(vector_expression<VecV> const& v,matrix_expression<MatA> const& A) {
	//compute it using the identity (v^TA)^T= A^Tv
	return matrix_vector_prod<matrix_transpose<MatA>,VecV>(trans(A),v());
}

//matrix-matrix prod
template<class MatA, class MatB>
class matrix_matrix_prod: public matrix_expression<matrix_matrix_prod<MatA, MatB> > {
public:
	typedef typename MatA::const_closure_type matrix_closure_typeA;
	typedef typename MatB::const_closure_type matrix_closure_typeB;
public:
	typedef decltype(
		typename MatA::scalar_type() * typename MatB::scalar_type()
	) scalar_type;
	typedef scalar_type value_type;
	typedef typename MatA::size_type size_type;
	typedef typename MatA::difference_type difference_type;
	typedef typename MatA::index_type index_type;
	typedef typename MatA::index_pointer index_pointer;
	typedef typename MatA::const_index_pointer const_index_pointer;

	typedef matrix_matrix_prod<MatA, MatB> const_closure_type;
	typedef const_closure_type closure_type;
	typedef unknown_storage_tag storage_category;
	typedef blockwise_tag evaluation_category;
	typedef unknown_orientation orientation;


	//FIXME: This workaround is required to be able to generate
	// temporary matrices
	typedef typename MatA::const_row_iterator const_row_iterator;
	typedef typename MatA::const_column_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	// Construction and destruction
	matrix_matrix_prod(
		matrix_closure_typeA const& matrixA,
		matrix_closure_typeB const& matrixB
	):m_matrixA(matrixA), m_matrixB(matrixB) {}

	size_type size1() const {
		return m_matrixA.size1();
	}
	size_type size2() const {
		return m_matrixB.size2();
	}
	
	matrix_closure_typeA const& matrixA() const {
		return m_matrixA;
	}
	matrix_closure_typeB const& matrixB() const {
		return m_matrixB;
	}
	
	//computation kernels
	template<class MatX>
	void assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		X().clear();
		plus_assign_to(X,alpha);
	}
	template<class MatX>
	void plus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		kernels::gemm(eval_block(m_matrixA), eval_block(m_matrixB), X, alpha);
	}
	
	template<class MatX>
	void minus_assign_to(matrix_expression<MatX>& X, scalar_type alpha = scalar_type(1) )const{
		kernels::gemm(eval_block(m_matrixA), eval_block(m_matrixB), X, -alpha);
	}
	
private:
	matrix_closure_typeA m_matrixA;
	matrix_closure_typeB m_matrixB;
};

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB>
matrix_matrix_prod<MatA,MatB> prod(
	matrix_expression<MatA> const& A,
	matrix_expression<MatB> const& B
) {
	return matrix_matrix_prod<MatA,MatB>(A(),B());
}

namespace detail{
	
template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB, column_major){
	for(std::size_t i = 0; i != matA.size2(); ++i){ 
		vecB(i)+=sum(column(matA,i));
	}
}

template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB, row_major){
	for(std::size_t i = 0; i != matA.size1(); ++i){ 
		noalias(vecB) += row(matA,i);
	}
}

template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB, unknown_orientation){
	sum_rows_impl(matA,vecB,row_major());
}

//dispatcher for triangular matrix
template<class MatA,class VecB,class Orientation,class Triangular>
void sum_rows_impl(MatA const& matA, VecB& vecB, packed<Orientation,Triangular>){
	sum_rows_impl(matA,vecB,Orientation());
}


//dispatcher 
template<class MatA,class VecB>
void sum_rows_impl(MatA const& matA, VecB& vecB){
	sum_rows_impl(matA,vecB,typename MatA::orientation());
}

template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA, column_major){
	typename MatA::value_type totalSum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		totalSum += sum(column(matA,j));
	}
	return totalSum;
}

template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA, row_major){
	typename MatA::value_type totalSum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		totalSum += sum(row(matA,i));
	}
	return totalSum;
}

template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA, unknown_orientation){
	return sum_impl(matA,row_major());
}


//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type sum_impl(MatA const& matA, packed<Orientation,Triangular>){
	return sum_impl(matA,Orientation());
}

//dispatcher
template<class MatA>
typename MatA::value_type sum_impl(MatA const& matA){
	return sum_impl(matA,typename MatA::orientation());
}

template<class MatA>
typename MatA::value_type max_impl(MatA const& matA, column_major){
	typename MatA::value_type maximum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		maximum = std::max(maximum, max(column(matA,j)));
	}
	return maximum;
}

template<class MatA>
typename MatA::value_type max_impl(MatA const& matA, row_major){
	typename MatA::value_type maximum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		maximum= std::max(maximum, max(row(matA,i)));
	}
	return maximum;
}

template<class MatA>
typename MatA::value_type max_impl(MatA const& matA, unknown_orientation){
	return max_impl(matA,row_major());
}

//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type max_impl(MatA const& matA, packed<Orientation,Triangular>){
	return std::max(max_impl(matA,Orientation()),0.0);
}

//dispatcher
template<class MatA>
typename MatA::value_type max_impl(MatA const& matA){
	return max_impl(matA,typename MatA::orientation());
}

template<class MatA>
typename MatA::value_type min_impl(MatA const& matA, column_major){
	typename MatA::value_type minimum = 0;
	for(std::size_t j = 0; j != matA.size2(); ++j){
		minimum= std::min(minimum, min(column(matA,j)));
	}
	return minimum;
}

template<class MatA>
typename MatA::value_type min_impl(MatA const& matA, row_major){
	typename MatA::value_type minimum = 0;
	for(std::size_t i = 0; i != matA.size1(); ++i){
		minimum= std::min(minimum, min(row(matA,i)));
	}
	return minimum;
}

template<class MatA>
typename MatA::value_type min_impl(MatA const& matA, unknown_orientation){
	return min_impl(matA,row_major());
}

//dispatcher for triangular matrix
template<class MatA,class Orientation,class Triangular>
typename MatA::value_type min_impl(MatA const& matA, packed<Orientation,Triangular>){
	return std::min(min_impl(matA,Orientation()),0.0);
}

//dispatcher
template<class MatA>
typename MatA::value_type min_impl(MatA const& matA){
	return min_impl(matA,typename MatA::orientation());
}

}//end detail

//dispatcher
template<class MatA>
typename vector_temporary_type<
	typename MatA::value_type,
	dense_random_access_iterator_tag
>::type
sum_rows(matrix_expression<MatA> const& A){
	typename vector_temporary_type<
		typename MatA::value_type,
		dense_random_access_iterator_tag
	>::type result(A().size2(),0.0);
	detail::sum_rows_impl(eval_block(A),result);
	return result;
}

template<class MatA>
typename vector_temporary_type<
	typename MatA::value_type,
	dense_random_access_iterator_tag
>::type
sum_columns(matrix_expression<MatA> const& A){
	//implemented using sum_rows_impl by transposing A
	typename vector_temporary_type<
		typename MatA::value_type,
		dense_random_access_iterator_tag
	>::type result(A().size1(),0);
	detail::sum_rows_impl(eval_block(trans(A)),result);
	return result;
}


template<class MatA>
typename MatA::value_type sum(matrix_expression<MatA> const& A){
	return detail::sum_impl(eval_block(A));
}

template<class MatA>
typename MatA::value_type max(matrix_expression<MatA> const& A){
	return detail::max_impl(eval_block(A));
}

template<class MatA>
typename MatA::value_type min(matrix_expression<MatA> const& A){
	return detail::min_impl(eval_block(A));
}

/// \brief Returns the frobenius inner-product between matrices exprssions 1 and e2.
///
///The frobenius inner product is defined as \f$ <A,B>_F=\sum_{ij} A_ij*B_{ij} \f$. It induces the
/// Frobenius norm \f$ ||A||_F = \sqrt{<A,A>_F} \f$
template<class E1, class E2>
decltype(typename E1::value_type() * typename E2::value_type())
frobenius_prod(
	matrix_expression<E1> const& e1,
	matrix_expression<E2> const& e2
) {
	return sum(eval_block(e1)*eval_block(e2));
}


template<class E>
typename matrix_norm_1<E>::result_type
norm_1(const matrix_expression<E> &e) {
	return matrix_norm_1<E>::apply(eval_block(e));
}

template<class E>
typename real_traits<typename E::value_type>::type
norm_frobenius(const matrix_expression<E> &e) {
	using std::sqrt;
	return sqrt(sum(abs_sqr(eval_block(e))));
}

template<class E>
typename matrix_norm_inf<E>::result_type
norm_inf(const matrix_expression<E> &e) {
	return matrix_norm_inf<E>::apply(eval_block(e));
}

/*!
 *  \brief Evaluates the sum of the values at the diagonal of
 *         matrix "v".
 *
 *  Example:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          {\bf 1} & 5       & 9        & 13\\
 *          2       & {\bf 6} & 10       & 14\\
 *          3       & 7       & {\bf 11} & 15\\
 *          4       & 8       & 12       & {\bf 16}\\
 *      \end{array}
 *      \right)
 *      \longrightarrow 1 + 6 + 11 + 16 = 34
 *  \f]
 *
 *      \param  m square matrix
 *      \return the sum of the values at the diagonal of \em m
 */
template < class MatrixT >
typename MatrixT::value_type trace(matrix_expression<MatrixT> const& m)
{
	SIZE_CHECK(m().size1() == m().size2());

	typename MatrixT::value_type t(m()(0, 0));
	for (unsigned i = 1; i < m().size1(); ++i)
		t += m()(i, i);
	return t;
}

/// \brief An diagonal matrix with values stored inside a diagonal vector
///
/// the matrix stores a Vector representing the diagonal.
template<class VectorType>
class diagonal_matrix: public matrix_expression<diagonal_matrix< VectorType > > {
	typedef diagonal_matrix< VectorType > self_type;
public:
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;
	typedef typename VectorType::value_type value_type;
	typedef typename VectorType::scalar_type scalar_type;
	typedef typename VectorType::const_reference const_reference;
	typedef typename VectorType::reference reference;
	typedef typename VectorType::pointer pointer;
	typedef typename VectorType::const_pointer const_pointer;

	typedef std::size_t index_type;
	typedef index_type const* const_index_pointer;
	typedef index_type index_pointer;

	typedef const matrix_reference<const self_type> const_closure_type;
	typedef matrix_reference<self_type> closure_type;
	typedef sparse_tag storage_category;
	typedef elementwise_tag evaluation_category;
	typedef row_major orientation;

	// Construction and destruction
	diagonal_matrix():m_zero(){}
	diagonal_matrix(VectorType const& diagonal):m_zero(),m_diagonal(diagonal){}

	// Accessors
	size_type size1() const {
		return m_diagonal.size();
	}
	size_type size2() const {
		return m_diagonal.size();
	}
	
	// Element access
	const_reference operator()(index_type i, index_type j) const {
		if (i == j)
			return m_diagonal(i);
		else
			return m_zero;
	}

	void set_element(size_type i, size_type j,value_type t){
		RANGE_CHECK(i == j);
		m_diagonal(i) = t;
	}

	// Assignment
	diagonal_matrix& operator = (diagonal_matrix const& m) {
		m_diagonal = m.m_diagonal;
		return *this;
	}

	// Swapping
	void swap(diagonal_matrix& m) {
		swap(m_diagonal,m.m_diagonal);
	}
	friend void swap(diagonal_matrix& m1, diagonal_matrix& m2) {
		m1.swap(m2);
	}
	
	//Iterators
	
	class const_row_iterator:public bidirectional_iterator_base<const_row_iterator, value_type> {
	public:
		typedef typename diagonal_matrix::value_type value_type;
		typedef typename diagonal_matrix::difference_type difference_type;
		typedef typename diagonal_matrix::const_reference reference;
		typedef value_type const* pointer;

		// Construction and destruction
		const_row_iterator(){}
		const_row_iterator(index_type index, value_type value, bool isEnd)
			:m_index(index),m_value(value),m_isEnd(isEnd){}

		// Arithmetic
		const_row_iterator& operator ++ () {
			m_isEnd = true;
			return *this;
		}
		const_row_iterator& operator -- () {
			m_isEnd = false;
			return *this;
		}

		// Dereference
		const_reference operator*() const {
			return m_value;
		}

		// Indices
		index_type index() const{
			return m_index;
		}

		// Assignment
		const_row_iterator& operator = (const_row_iterator const& it) {
			m_index = it.m_index;
			return *this;
		}

		// Comparison
		bool operator == (const_row_iterator const& it) const {
			RANGE_CHECK(m_index == it.m_index);
			return m_isEnd == it.m_isEnd;
		}

	private:
		index_type m_index;
		value_type m_value;
		bool m_isEnd;
	};
	typedef const_row_iterator const_column_iterator;
	typedef const_row_iterator row_iterator;
	typedef const_column_iterator column_iterator;

	
	const_row_iterator row_begin(index_type i) const {
		return row_iterator(i, m_diagonal(i),false);
	}
	const_row_iterator row_end(index_type i) const {
		return const_row_iterator(i, m_zero,true);
	}
	const_column_iterator column_begin(index_type i) const {
		return column_iterator(i, m_diagonal(i),false);
	}
	const_column_iterator column_end(index_type i) const {
		return const_column_iterator(i, m_zero,true);
	}

private:
	value_type const m_zero;
	VectorType m_diagonal; 
};

/** \brief An identity matrix with values of type \c T
 *
 * Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
 */
template<class T>
class identity_matrix: public diagonal_matrix<scalar_vector<T> > {
	typedef diagonal_matrix<scalar_vector<T> > base_type;
public:
	identity_matrix(){}
	identity_matrix(std::size_t size):base_type(scalar_vector<T>(size,T(1))){}
};

}
}

#endif
