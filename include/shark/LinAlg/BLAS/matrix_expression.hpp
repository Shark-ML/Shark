/*!
 * \brief       Expression templates for expressions involving matrices
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
#ifndef SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_MATRIX_EXPRESSION_HPP

#include "detail/matrix_expression_classes.hpp"
#include "detail/matrix_expression_optimizers.hpp"
#include <boost/utility/enable_if.hpp>


namespace shark {
namespace blas {


///\brief Computes the outer product of two vectors.
///
/// The outer product of two vectors v1 and v2 is defined as the matrix
/// (outer_prod (v1, v2))_ij [i] [j] = v1[i] * v2 [j]
template<class E1, class E2>
outer_product<E1, E2 >
outer_prod(
	vector_expression<E1> const& e1,
        vector_expression<E2> const& e2
) {
	return outer_product<E1, E2>(e1(), e2());
}



///\brief Creates a matrix from a vector by repeating the vector in every row of the matrix.
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

/// \brief Repeats a single element to form a matrix  of size rows x columns.
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_matrix<T> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T>(rows, columns, scalar);
}


/// \brief Computes the multiplication of a matrix-expression e with a scalar t.
///
/// \f$ (e*t)_{ij} = e_{ij}*t \f$
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (matrix_expression<E> const& e, T scalar){
	return matrix_scalar_multiply<E>(e(), typename E::scalar_type(scalar));
}

/// \brief Computes the multiplication of a matrix-expression e with a scalar t.
///
/// \f$ (t*e)_{ij} = t*e_{ij} \f$
template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        matrix_scalar_multiply<E> 
>::type
operator* (T scalar, matrix_expression<E> const& e){
	return matrix_scalar_multiply<E>(e(), typename E::scalar_type(scalar));
}

/// \brief Negates the matrix-expression e.
///
/// \f$ (-e)_{ij} = - e_{ij} \f$
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


/// \brief computes the matrix-vector product x+=Av
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV>
typename detail::matrix_vector_prod_optimizer<MatA,VecV>::type prod(matrix_expression<MatA> const& A,vector_expression<VecV> const& v) {
	return detail::matrix_vector_prod_optimizer<MatA,VecV>::create(A(),v());
}

/// \brief computes the matrix-vector product x+=v^TA
///
/// it is computed via the identity (v^TA)^T= A^Tv
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV>
typename detail::matrix_vector_prod_optimizer<matrix_transpose<MatA>,VecV>::type prod(vector_expression<VecV> const& v,matrix_expression<MatA> const& A) {
	typedef typename matrix_transpose<MatA>::const_closure_type closure;
	return detail::matrix_vector_prod_optimizer<matrix_transpose<MatA>,VecV>::create(closure(A()),v());
}

/// \brief Computes the matrix-vector product x+= alpha * Av or x= alpha * Av
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class VecV>
matrix_vector_prod<detail::dense_triangular_proxy<MatA const,TriangularType> ,VecV> triangular_prod(
	matrix_expression<MatA> const& A,
	vector_expression<VecV>& v
) {
	typedef detail::dense_triangular_proxy<MatA const,TriangularType> Wrapper;
	return matrix_vector_prod<Wrapper ,VecV>(Wrapper(A()), v());
}

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB>
matrix_matrix_prod<MatA,MatB> prod(
	matrix_expression<MatA> const& A,
	matrix_expression<MatB> const& B
) {
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	return matrix_matrix_prod<MatA,MatB>(A(),B());
}

/// \brief Computes the matrix-vector product x+= alpha * AB or x= alpha * AB
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
/// B is interpreted as dense matrix.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class MatB>
matrix_matrix_prod<detail::dense_triangular_proxy<MatA const,TriangularType> ,MatB>
triangular_prod(
	matrix_expression<MatA> const& A,
	matrix_expression<MatB> const& B
) {
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	typedef detail::dense_triangular_proxy<MatA const,TriangularType> Wrapper;
	return matrix_matrix_prod<Wrapper ,MatB>(Wrapper(A()), B());
}

namespace detail{

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
typename MatA::value_type sum_impl(MatA const& matA, triangular<Orientation,Triangular>){
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
typename MatA::value_type max_impl(MatA const& matA, triangular<Orientation, Triangular>){
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
typename MatA::value_type min_impl(MatA const& matA, triangular<Orientation,Triangular>){
	return std::min(min_impl(matA,Orientation()),0.0);
}

//dispatcher
template<class MatA>
typename MatA::value_type min_impl(MatA const& matA){
	return min_impl(matA,typename MatA::orientation());
}

}//end detail


template<class MatA>
sum_matrix_rows<MatA>
sum_rows(matrix_expression<MatA> const& A){
	return sum_matrix_rows<MatA>(A());
}

template<class MatA>
sum_matrix_rows<typename detail::matrix_transpose_optimizer<typename const_expression<MatA>::type >::type >
sum_columns(matrix_expression<MatA> const& A){
	return sum_rows(trans(A));
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

/// \brief Computes the matrix 1-norm |A|_1
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class E>
typename real_traits<typename E::value_type>::type
norm_1(matrix_expression<E> const& e) {
	return max(sum_rows(abs(e)));
}

/// \brief computes the frobenius norm |A|_F
///
/// It is defined as \f$ Tr(A^TA)=\sum_{ij} A_{ij}^2 \f$
template<class E>
typename real_traits<typename E::value_type>::type
norm_frobenius(matrix_expression<E> const& e) {
	using std::sqrt;
	return sqrt(sum(abs_sqr(eval_block(e))));
}

/// \brief Computes the matrix inf-norm |A|_inf
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class E>
typename real_traits<typename E::value_type>::type
norm_inf(matrix_expression<E> const& e) {
	return max(sum_columns(abs(e)));
}

/// \brief Evaluates the trace of matrix m
///
/// The rtace is defined as the sum of the diagonal elements of m,
/// \f$ \text{trace}(m) = \sum_i m_{ii}\f$
///
/// \param  m square matrix
/// \return the sum of the values at the diagonal of \em m
template < class MatrixT >
typename MatrixT::value_type trace(matrix_expression<MatrixT> const& m)
{
	SIZE_CHECK(m().size1() == m().size2());
	return sum(diag(m));
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
