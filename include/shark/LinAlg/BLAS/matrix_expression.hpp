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
#ifndef REMORA_MATRIX_EXPRESSION_HPP
#define REMORA_MATRIX_EXPRESSION_HPP

#include "detail/expression_optimizers.hpp"
#include <boost/utility/enable_if.hpp>
#include "kernels/matrix_fold.hpp"
#include "matrix_proxy.hpp"
#include "vector_proxy.hpp"
#include "vector_expression.hpp"

namespace remora{


///\brief Computes the outer product of two vectors.
///
/// The outer product of two vectors v1 and v2 is defined as the matrix
/// (outer_prod (v1, v2))_ij [i] [j] = v1[i] * v2 [j]
template<class VecA, class VecB, class Device>
outer_product<VecA, VecB >
outer_prod(
	vector_expression<VecA, Device> const& v1,
        vector_expression<VecB, Device> const& v2
) {
	return outer_product<VecA, VecB>(v1(), v2());
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
template<class VecV, class Device>
vector_repeater<VecV> repeat(vector_expression<VecV, Device> const& vector, std::size_t rows){
	return vector_repeater<VecV>(vector(),rows);
}

/// \brief Repeats a single element to form a matrix  of size rows x columns.
///
///TODO: cpu only!
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_matrix<T,cpu_tag> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T,cpu_tag>(rows, columns, scalar);
}


/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (A*t)_{ij} = e_{ij}*t \f$
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type >,
        matrix_scalar_multiply<MatA> 
>::type
operator* (matrix_expression<MatA, Device> const& A, T scalar){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::value_type(scalar));
}

/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (t*A)_{ij} = t*e_{ij} \f$
template<class T, class MatA, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type >,
        matrix_scalar_multiply<MatA> 
>::type
operator* (T scalar, matrix_expression<MatA, Device> const& A){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::value_type(scalar));
}

/// \brief Negates the matrix-expression A.
///
/// \f$ (-A)_{ij} = - e_{ij} \f$
template<class MatA, class Device>
matrix_scalar_multiply<MatA> operator-(matrix_expression<MatA, Device> const& A){
	return matrix_scalar_multiply<MatA>(A(), typename MatA::value_type(-1));
}

#define REMORA_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class MatA, class Device>\
matrix_unary<MatA,typename device_traits<Device>:: template F<typename MatA::value_type> >\
name(matrix_expression<MatA, Device> const& v){\
	typedef typename device_traits<Device>:: template F<typename MatA::value_type> functor_type;\
	return matrix_unary<MatA, functor_type >(v(), functor_type());\
}
REMORA_UNARY_MATRIX_TRANSFORMATION(abs, abs)
REMORA_UNARY_MATRIX_TRANSFORMATION(log, log)
REMORA_UNARY_MATRIX_TRANSFORMATION(exp, exp)
REMORA_UNARY_MATRIX_TRANSFORMATION(tanh,tanh)
REMORA_UNARY_MATRIX_TRANSFORMATION(sqr, sqr)
REMORA_UNARY_MATRIX_TRANSFORMATION(sqrt, sqrt)
REMORA_UNARY_MATRIX_TRANSFORMATION(sigmoid, sigmoid)
REMORA_UNARY_MATRIX_TRANSFORMATION(softPlus, soft_plus)
REMORA_UNARY_MATRIX_TRANSFORMATION(elem_inv, inv)
#undef REMORA_UNARY_MATRIX_TRANSFORMATION

///\brief Adds two Matrices
template<class MatA, class MatB, class Device>
matrix_addition<MatA, MatB > operator+ (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return matrix_addition<MatA, MatB>(A(),B());
}

///\brief Subtracts two Matrices
template<class MatA, class MatB, class Device>
matrix_addition<MatA, matrix_scalar_multiply<MatB> > operator- (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return matrix_addition<MatA, matrix_scalar_multiply<MatB> >(A(),-B());
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type>, 
	matrix_addition<MatA, scalar_matrix<T,Device> >
>::type operator+ (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A + scalar_matrix<T,Device>(A().size1(),A().size2(),t);
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class T, class MatA, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type>,
	matrix_addition<MatA, scalar_matrix<T,Device> >
>::type operator+ (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return A + scalar_matrix<T,Device>(A().size1(),A().size2(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant matrix from a matrix.
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type> ,
	matrix_addition<MatA, matrix_scalar_multiply<scalar_matrix<T,Device> > >
>::type operator- (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A - scalar_matrix<T,Device>(A().size1(),A().size2(),t);
}

///\brief Subtracts a matrix from a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename boost::enable_if<
	std::is_convertible<T, typename MatA::value_type>,
	matrix_addition<scalar_matrix<T,Device>, matrix_scalar_multiply<MatA> >
>::type operator- (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return scalar_matrix<T,Device>(A().size1(),A().size2(),t) - A;
}

#define REMORA_BINARY_MATRIX_EXPRESSION(name, F)\
template<class MatA, class MatB, class Device>\
matrix_binary<MatA, MatB, typename device_traits<Device>:: template F<typename common_value_type<MatA,MatB>::type> >\
name(matrix_expression<MatA, Device> const& m1, matrix_expression<MatB, Device> const& m2){\
	SIZE_CHECK(m1().size1() == m2().size1());\
	SIZE_CHECK(m1().size2() == m2().size2());\
	typedef typename common_value_type<MatA,MatB>::type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return matrix_binary<MatA, MatB, functor_type >(m1(),m2(), functor_type());\
}
REMORA_BINARY_MATRIX_EXPRESSION(operator*, multiply)
REMORA_BINARY_MATRIX_EXPRESSION(element_prod, multiply)
REMORA_BINARY_MATRIX_EXPRESSION(operator/, divide)
REMORA_BINARY_MATRIX_EXPRESSION(element_div, divide)
REMORA_BINARY_MATRIX_EXPRESSION(pow,pow)
REMORA_BINARY_MATRIX_EXPRESSION(min,min)
REMORA_BINARY_MATRIX_EXPRESSION(max,max)
#undef REMORA_BINARY_MATRIX_EXPRESSION

#define REMORA_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class T, class MatA, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename MatA::value_type >,\
        matrix_binary<MatA, scalar_matrix<typename MatA::value_type,Device>,typename device_traits<Device>:: template  F<typename MatA::value_type> > \
>::type \
name (matrix_expression<MatA, Device> const& m, T t){ \
	typedef typename MatA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return matrix_binary<MatA, scalar_matrix<type,Device>, functor_type >(m(), scalar_matrix<type,Device>(m().size1(), m().size2(), t) ,functor_type()); \
}
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator/, divide)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator<, less_than)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator<=, less_equal_than)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator>, bigger_than)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator>=, bigger_equal_than)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator==, equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator!=, not_equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(min, min)
REMORA_MATRIX_SCALAR_TRANSFORMATION(max, max)
REMORA_MATRIX_SCALAR_TRANSFORMATION(pow, pow)
#undef REMORA_MATRIX_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define REMORA_MATRIX_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class MatA, class Device> \
typename boost::enable_if< \
	std::is_convertible<T, typename MatA::value_type >,\
	matrix_binary<scalar_matrix< typename MatA::value_type,Device>, MatA, typename device_traits<Device>:: template F< typename MatA::value_type> > \
>::type \
name (T t, matrix_expression<MatA, Device> const& m){ \
	typedef typename MatA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return  matrix_binary<scalar_matrix<type,Device>, MatA, functor_type >(scalar_matrix<type,Device>(m().size1(), m().size2(), t), m(), functor_type()); \
}
REMORA_MATRIX_SCALAR_TRANSFORMATION_2(min, min)
REMORA_MATRIX_SCALAR_TRANSFORMATION_2(max, max)
#undef REMORA_MATRIX_SCALAR_TRANSFORMATION_2

template<class MatA, class MatB, class Device>
matrix_binary<MatA, MatB, 
	typename device_traits<Device>:: template  safe_divide<typename common_value_type<MatA,MatB>::type> 
>
safe_div(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device> const& B, 
	typename common_value_type<MatA,MatB>::type defaultValue
){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typedef typename common_value_type<MatA,MatB>::type result_type;
	typedef typename device_traits<Device>:: template safe_divide<result_type> functor_type;
	return matrix_binary<MatA, MatB, functor_type>(A(),B(), functor_type(defaultValue));
}


/// \brief computes the matrix-vector product x+=Av
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV, class Device>
typename detail::matrix_vector_prod_optimizer<MatA,VecV>::type prod(
	matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v
) {
	SIZE_CHECK(A().size2() == v().size());
	return detail::matrix_vector_prod_optimizer<MatA,VecV>::create(A(),v());
}

/// \brief computes the matrix-vector product x+=v^TA
///
/// it is computed via the identity (v^TA)^T= A^Tv
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV, class Device>
auto prod(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A) -> decltype(prod(trans(A),v)){
	SIZE_CHECK(A().size1() == v().size());
	return prod(trans(A),v);
}

/// \brief Operator syntax for computes the matrix-vector product
///
/// v%A= prod(v,A).
template<class MatA, class VecV, class Device>
auto operator%(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A) -> decltype(prod(trans(A),v)){
	SIZE_CHECK(A().size1() == v().size());
	return prod(trans(A),v);
}

/// \brief Operator syntax for computes the matrix-vector product
///
/// A%v = prod(A,v).
template<class MatA, class VecV, class Device>
auto operator%(matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v) -> decltype(prod(A,v)){
	SIZE_CHECK(A().size2() == v().size());
	return prod(A,v);
}

/// \brief Computes the matrix-vector product x+= alpha * Av or x= alpha * Av
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class VecV, class Device>
matrix_vector_prod<detail::dense_triangular_proxy<MatA const,TriangularType> ,VecV> triangular_prod(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device>& v
) {
	SIZE_CHECK(A().size2() == v().size());
	typedef detail::dense_triangular_proxy<MatA const,TriangularType> Wrapper;
	return matrix_vector_prod<Wrapper ,VecV>(Wrapper(A()), v());
}

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB, class Device>
typename detail::matrix_matrix_prod_optimizer<MatA,MatB>::type prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size2() == B().size1());
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	return detail::matrix_matrix_prod_optimizer<MatA,MatB>::create(A(),B());
}

/// \brief Operator syntax for computes the matrix-matrix product
///
/// A%B= prod(A,B).
template<class MatA, class MatB, class Device>
auto operator%(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) -> decltype(prod(A,B)){
	SIZE_CHECK(A().size2() == B().size1());
	return prod(A,B);
}

/// \brief Computes the matrix-vector product x+= alpha * AB or x= alpha * AB
///
/// A is interpreted as triangular matrix.
/// The first template argument governs the type
/// of triangular matrix: lower, upper, unit_lower and unit_upper.
/// B is interpreted as dense matrix.
///
///Example: x += triangular_prod<lower>(A,v);
template<class TriangularType, class MatA, class MatB, class Device>
matrix_matrix_prod<detail::dense_triangular_proxy<typename const_expression<MatA>::type,TriangularType> ,MatB>
triangular_prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size2() == B().size1());
	static_assert(std::is_base_of<linear_structure, typename MatA::orientation>::value, "A must be linearly stored");
	static_assert(std::is_base_of<linear_structure, typename MatB::orientation>::value, "B must be linearly stored");
	typedef detail::dense_triangular_proxy<typename const_expression<MatA>::type,TriangularType> Wrapper;
	return matrix_matrix_prod<Wrapper ,MatB>(Wrapper(A()), B());
}

template<class MatA, class Device>
sum_matrix_rows<MatA>
sum_rows(matrix_expression<MatA, Device> const& A){
	return sum_matrix_rows<MatA>(A());
}

template<class MatA, class Device>
auto sum_columns(matrix_expression<MatA, Device> const& A)->decltype(sum_rows(trans(A))){
	return sum_rows(trans(A));
}


template<class MatA, class Device>
typename MatA::value_type sum(matrix_expression<MatA, Device> const& A){
	typedef typename MatA::value_type value_type;
	typedef typename device_traits<Device>:: template add<value_type> functor_type;
	auto const& elem_result = eval_block(A);
	value_type result = 0;
	kernels::matrix_fold<functor_type>(elem_result,result);
	return result;
}

template<class MatA, class Device>
typename MatA::value_type max(matrix_expression<MatA, Device> const& A){
	typedef typename MatA::value_type value_type;
	typedef typename device_traits<Device>:: template max<value_type> functor_type;
	auto const& elem_result = eval_block(A);
	value_type result = 0;
	kernels::matrix_fold<functor_type>(elem_result,result);
	return result;
}

template<class MatA, class Device>
typename MatA::value_type min(matrix_expression<MatA, Device> const& A){
	typedef typename MatA::value_type value_type;
	typedef typename device_traits<Device>:: template min<value_type> functor_type;
	auto const& elem_result = eval_block(A);
	value_type result = 0;
	kernels::matrix_fold<functor_type>(elem_result,result);
	return result;
}

/// \brief Returns the frobenius inner-product between matrices exprssions 1 and B.
///
///The frobenius inner product is defined as \f$ <A,B>_F=\sum_{ij} A_ij*B_{ij} \f$. It induces the
/// Frobenius norm \f$ ||A||_F = \sqrt{<A,A>_F} \f$
template<class MatA, class MatB, class Device>
decltype(typename MatA::value_type() * typename MatB::value_type())
frobenius_prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	return sum(eval_block(A*B));
}

/// \brief Computes the matrix 1-norm |A|_1
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class MatA, class Device>
typename real_traits<typename MatA::value_type>::type
norm_1(matrix_expression<MatA, Device> const& A) {
	return max(sum_rows(abs(A)));
}

/// \brief computes the frobenius norm |A|_F
///
/// It is defined as \f$ \sqrt{Tr(A^TA)}=\sqrt{\sum_{ij} A_{ij}^2} \f$
template<class MatA, class Device>
typename real_traits<typename MatA::value_type>::type
norm_frobenius(matrix_expression<MatA, Device> const& A) {
	using std::sqrt;
	return sqrt(sum(sqr(eval_block(A))));
}

/// \brief Computes the matrix inf-norm |A|_inf
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class MatA, class Device>
typename real_traits<typename MatA::value_type>::type
norm_inf(matrix_expression<MatA, Device> const& A) {
	return max(sum_columns(abs(A)));
}

/// \brief Evaluates the trace of matrix A
///
/// The rtace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
///
/// \param  A square matrix
/// \return the sum of the values at the diagonal of \em A
template < class MatA, class Device>
typename MatA::value_type trace(matrix_expression<MatA, Device> const& A)
{
	SIZE_CHECK(A().size1() == A().size2());
	return sum(diag(A));
}

/** \brief An identity matrix with values of type \c T
 *
 * Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
 */
template<class T>
class identity_matrix: public diagonal_matrix<scalar_vector<T, cpu_tag> > {
	typedef diagonal_matrix<scalar_vector<T, cpu_tag> > base_type;
public:
	identity_matrix(){}
	identity_matrix(std::size_t size):base_type(scalar_vector<T, cpu_tag>(size,T(1))){}
};


template<class MatA, class Device>
diagonal_matrix<MatA> to_diagonal(vector_expression<MatA, Device> const& A){
	return diagonal_matrix<MatA>(A());
}

}

#endif
