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
#include "kernels/matrix_fold.hpp"
#include "proxy_expressions.hpp"
#include "vector_expression.hpp"
//~ #include "vector_set_expressions.hpp"

namespace remora{
	
/////////////////////////////////////////////
//////////Vector->Matrix Operations
/////////////////////////////////////////////


///\brief Creates a matrix from a vector by repeating the vector in every row of the matrix.
///
///TODO: cpu only!
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
typename std::enable_if<std::is_arithmetic<T>::value, scalar_matrix<T,cpu_tag, row_major> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T,cpu_tag, row_major>(rows, columns, scalar);
}

/// \brief An identity matrix with values of type \c T
/// 
/// Elements or cordinates \f$(i,i)\f$ are equal to 1 (one) and all others to 0 (zero).
template<class T>
class identity_matrix: public diagonal_matrix<scalar_vector<T, cpu_tag> > {
	typedef diagonal_matrix<scalar_vector<T, cpu_tag> > base_type;
public:
	identity_matrix(){}
	identity_matrix(std::size_t size):base_type(scalar_vector<T, cpu_tag>(size,T(1))){}
};



/// \brief Creates a nxn diagonal matrix with with diagonal given by a vector.
template<class MatA, class Device>
diagonal_matrix<MatA> to_diagonal(vector_expression<MatA, Device> const& v){
	return diagonal_matrix<MatA>(v());
}

/////////////////////////////////////////////
//////////Unary Matrix Transformations
/////////////////////////////////////////////


/// \brief Negates the matrix-expression A.
///
/// \f$ (-A)_{ij} = - e_{ij} \f$
template<class MatA, class Device>
typename detail::matrix_scalar_multiply_optimizer<MatA>::type
operator-(matrix_expression<MatA, Device> const& A){
	return detail::matrix_scalar_multiply_optimizer<MatA>::create(A(), typename MatA::value_type(-1));
}

#define REMORA_UNARY_MATRIX_TRANSFORMATION(name, F)\
template<class MatA, class Device>\
typename detail::matrix_unary_optimizer<MatA,typename device_traits<Device>:: template F<typename MatA::value_type> >::type \
name(matrix_expression<MatA, Device> const& m){\
	typedef typename device_traits<Device>:: template F<typename MatA::value_type> functor_type;\
	return detail::matrix_unary_optimizer<MatA, functor_type >::create(m(), functor_type());\
}

REMORA_UNARY_MATRIX_TRANSFORMATION(abs, abs)
REMORA_UNARY_MATRIX_TRANSFORMATION(log, log)
REMORA_UNARY_MATRIX_TRANSFORMATION(exp, exp)
REMORA_UNARY_MATRIX_TRANSFORMATION(tanh,tanh)
REMORA_UNARY_MATRIX_TRANSFORMATION(sin,sin)
REMORA_UNARY_MATRIX_TRANSFORMATION(cos,cos)
REMORA_UNARY_MATRIX_TRANSFORMATION(tan,tan)
REMORA_UNARY_MATRIX_TRANSFORMATION(asin,asin)
REMORA_UNARY_MATRIX_TRANSFORMATION(acos,acos)
REMORA_UNARY_MATRIX_TRANSFORMATION(atan,atan)
REMORA_UNARY_MATRIX_TRANSFORMATION(erf,erf)
REMORA_UNARY_MATRIX_TRANSFORMATION(erfc,erfc)
REMORA_UNARY_MATRIX_TRANSFORMATION(sqr, sqr)
REMORA_UNARY_MATRIX_TRANSFORMATION(sqrt, sqrt)
REMORA_UNARY_MATRIX_TRANSFORMATION(cbrt, cbrt)
REMORA_UNARY_MATRIX_TRANSFORMATION(sigmoid, sigmoid)
REMORA_UNARY_MATRIX_TRANSFORMATION(softPlus, soft_plus)
REMORA_UNARY_MATRIX_TRANSFORMATION(elem_inv, inv)
#undef REMORA_UNARY_MATRIX_TRANSFORMATION

#define REMORA_UNARY_VECTOR_SET_TRANSFORMATION(name)\
template<class S, class Device>\
auto name(vector_set_expression<S, Device> const& set)\
-> decltype(as_set(name( set().expression()), typename S::point_orientation())){\
	return as_set(name( set().expression()), typename S::point_orientation());\
}
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(abs)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(log)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(exp)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(tanh)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(sin)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(cos)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(tan)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(asin)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(acos)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(atan)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(erf)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(erfc)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(sqr)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(sqrt)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(cbrt)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(sigmoid)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(softPlus)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(elem_inv)
REMORA_UNARY_VECTOR_SET_TRANSFORMATION(operator-)
#undef REMORA_UNARY_VECTOR_SET_TRANSFORMATION


/////////////////////////////////////////////
//////////Matrix-Scalar Operations
/////////////////////////////////////////////

/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (A*t)_{ij} = e_{ij}*t \f$
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type >::value,
	typename detail::matrix_scalar_multiply_optimizer<MatA>::type
>::type
operator* (matrix_expression<MatA, Device> const& A, T scalar){
	return detail::matrix_scalar_multiply_optimizer<MatA>::create(A(), typename MatA::value_type(scalar));
}

/// \brief Computes the multiplication of a matrix-expression A with a scalar t.
///
/// \f$ (t*A)_{ij} = t*e_{ij} \f$
template<class T, class MatA, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type >::value,
        typename detail::matrix_scalar_multiply_optimizer<MatA>::type
>::type
operator* (T scalar, matrix_expression<MatA, Device> const& A){
	return detail::matrix_scalar_multiply_optimizer<MatA>::create(A(), typename MatA::value_type(scalar));
}


///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value, 
	matrix_addition<MatA, scalar_matrix<T,Device, typename MatA::orientation> >
>::type operator+ (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A + scalar_matrix<T,Device, typename MatA::orientation>(A().size1(),A().size2(),t);
}

///\brief Adds a matrix plus a scalar which is interpreted as a constant matrix
template<class T, class MatA, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value,
	matrix_addition<MatA, scalar_matrix<T,Device, typename MatA::orientation> >
>::type operator+ (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return A + scalar_matrix<T,Device, typename MatA::orientation>(A().size1(),A().size2(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant matrix from a matrix.
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value ,
	decltype(std::declval<MatA const&>() + T())
>::type operator- (
	matrix_expression<MatA, Device> const& A,
	T t
){
	return A + (-t);
}

///\brief Subtracts a matrix from a scalar which is interpreted as a constant matrix
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value,
	decltype(T() + (-std::declval<MatA const&>()))
>::type operator- (
	T t,
	matrix_expression<MatA, Device> const& A
){
	return t + (-A);
}

#define REMORA_MATRIX_SCALAR_TRANSFORMATION(name, F)\
template<class T, class MatA, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename MatA::value_type >::value,\
        matrix_binary<MatA, scalar_matrix<typename MatA::value_type,Device, typename MatA::orientation>,typename device_traits<Device>:: template  F<typename MatA::value_type> > \
>::type \
name (matrix_expression<MatA, Device> const& m, T t){ \
	typedef typename MatA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_matrix<type,Device, typename MatA::orientation> mat_type;\
	return matrix_binary<MatA, mat_type, functor_type >(m(), mat_type(m().size1(), m().size2(), t) ,functor_type()); \
}
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator/, divide)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator<, less)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator<=, less_equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator>, greater)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator>=, greater_equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator==, equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(operator!=, not_equal)
REMORA_MATRIX_SCALAR_TRANSFORMATION(min, min)
REMORA_MATRIX_SCALAR_TRANSFORMATION(max, max)
REMORA_MATRIX_SCALAR_TRANSFORMATION(pow, pow)
#undef REMORA_MATRIX_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i,j] = op(t,v[i,j])
#define REMORA_MATRIX_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class MatA, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename MatA::value_type >::value,\
	matrix_binary<scalar_matrix< typename MatA::value_type,Device, typename MatA::orientation>, MatA, typename device_traits<Device>:: template F< typename MatA::value_type> > \
>::type \
name (T t, matrix_expression<MatA, Device> const& m){ \
	typedef typename MatA::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	typedef scalar_matrix<type,Device, typename MatA::orientation> mat_type;\
	return  matrix_binary<mat_type, MatA, functor_type >(mat_type(m().size1(), m().size2(), t), m(), functor_type()); \
}
REMORA_MATRIX_SCALAR_TRANSFORMATION_2(min, min)
REMORA_MATRIX_SCALAR_TRANSFORMATION_2(max, max)
#undef REMORA_MATRIX_SCALAR_TRANSFORMATION_2


/////////////////////////////////////////////
//////////Simple Matrix-Binary Operations
/////////////////////////////////////////////


///\brief Adds two Matrices
template<class MatA, class MatB, class Device>
matrix_addition<MatA, MatB > operator+ (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	return matrix_addition<MatA, MatB>(A(),B());
}

///\brief Subtracts two Matrices
template<class MatA, class MatB, class Device>
auto operator- (
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) -> decltype(A() + (-B)){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	return A() + (-B);
}

template<class MatA, class MatB, class Device>
matrix_binary<MatA, MatB, 
	typename device_traits<Device>:: template  safe_divide<typename common_value_type<MatA,MatB>::type> 
>
safe_div(
	matrix_expression<MatA, Device> const& A, 
	matrix_expression<MatB, Device> const& B, 
	typename common_value_type<MatA,MatB>::type defaultValue
){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typedef typename common_value_type<MatA,MatB>::type result_type;
	typedef typename device_traits<Device>:: template safe_divide<result_type> functor_type;
	return matrix_binary<MatA, MatB, functor_type>(A(),B(), functor_type(defaultValue));
}


#define REMORA_BINARY_MATRIX_EXPRESSION(name, F)\
template<class MatA, class MatB, class Device>\
matrix_binary<MatA, MatB, typename device_traits<Device>:: template F<typename common_value_type<MatA,MatB>::type> >\
name(matrix_expression<MatA, Device> const& m1, matrix_expression<MatB, Device> const& m2){\
	REMORA_SIZE_CHECK(m1().size1() == m2().size1());\
	REMORA_SIZE_CHECK(m1().size2() == m2().size2());\
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

/////////////////////////////////////////
/////////Matrix-Products
/////////////////////////////////////////


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


/// \brief computes the matrix-vector product x+=Av
///
/// The call to prod does not compute the product itself but instead, as all other expressions,
/// it returns an expression-object which can compute it. In contrast to other expression,
/// this expression is optimized to make use of well known mathematical identities to reduce run time of the algorithm.
template<class MatA, class VecV, class Device>
typename detail::matrix_vector_prod_optimizer<MatA,VecV>::type prod(
	matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v
) {
	REMORA_SIZE_CHECK(A().size2() == v().size());
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
	REMORA_SIZE_CHECK(A().size1() == v().size());
	return prod(trans(A),v);
}

/// \brief Operator syntax for computes the matrix-vector product
///
/// v%A= prod(v,A).
template<class MatA, class VecV, class Device>
auto operator%(vector_expression<VecV, Device> const& v,matrix_expression<MatA, Device> const& A) -> decltype(prod(trans(A),v)){
	REMORA_SIZE_CHECK(A().size1() == v().size());
	return prod(trans(A),v);
}

/// \brief Operator syntax for computes the matrix-vector product
///
/// A%v = prod(A,v).
template<class MatA, class VecV, class Device>
auto operator%(matrix_expression<MatA, Device> const& A,vector_expression<VecV, Device> const& v) -> decltype(prod(A,v)){
	REMORA_SIZE_CHECK(A().size2() == v().size());
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
auto triangular_prod(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device>& v
) -> decltype(prod(to_triangular(A, TriangularType()), v)){
	REMORA_SIZE_CHECK(A().size2() == v().size());
	return prod(to_triangular(A, TriangularType()), v);
}

/// \brief computes the matrix-matrix product X+=AB
template<class MatA, class MatB, class Device>
typename detail::matrix_matrix_prod_optimizer<MatA,MatB>::type prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
) {
	REMORA_SIZE_CHECK(A().size2() == B().size1());
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
	REMORA_SIZE_CHECK(A().size2() == B().size1());
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
auto triangular_prod(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
)  -> decltype(prod(to_triangular(A, TriangularType()), B)){
	REMORA_SIZE_CHECK(A().size2() == B().size1());
	return prod(to_triangular(A, TriangularType()), B);
}


//~ // inner_prod (set, v)_i = inner_prod(set_i,v)
//~ template<class S, class V, class Device>
//~ typename detail::vector_set_inner_prod_optimizer<S,V>::type
//~ inner_prod(
	//~ vector_set_expression<S, Device> const& set,
	//~ vector_expression<V, Device> const& v
//~ ){
	//~ REMORA_SIZE_CHECK(set().point_size() == v().size());
	//~ return detail::vector_set_inner_prod_optimizer<S,V>::create(set,v);
//~ }

//~ template<class S, class V, class Device>
//~ typename detail::vector_set_inner_prod_optimizer<S,V>::type
//~ inner_prod(
	//~ vector_expression<V, Device> const& v
	//~ vector_set_expression<S, Device> const& set
//~ ){
	//~ REMORA_SIZE_CHECK(set1().point_size() == v().size());
	//~ return detail::vector_set_inner_prod_optimizer<S,V>::create(set,v);
//~ }

//~ template<class S, class M, class Device>
//~ typename detail::vector_set_matrix_prod_optimizer<S,M>::type
//~ operator%(
	//~ vector_set_expression<S, Device> const& set,
	//~ matrix_expression<M, Device> const& m
//~ ){
	//~ REMORA_SIZE_CHECK(set().point_size() == m().size1());
	//~ return detail::vector_set_matrix_prod_optimizer<S,M>::create(set,m);
//~ }

//~ template<class S, class M, class Device>
//~ auto operator%(
	//~ matrix_expression<M, Device> const& m
	//~ vector_set_expression<S, Device> const& set
//~ )->decltype(set % trans(m)){
	//~ REMORA_SIZE_CHECK(set().point_size() == m().size2());
	//~ return set % trans(m);
//~ }


/////////////////////////////////////////
//////////VECTOR-SET REDUCTIONS
/////////////////////////////////////////

/// \brief Computes the sum of elements for each point in the set
///
/// Formula: (sum S)_i = sum_j S_ij
template<class S, class Device>
typename detail::fold_vector_set_optimizer<
	S, typename device_traits<Device>:: template add<typename S::value_type>
	, typename device_traits<Device>:: template identity<typename S::value_type>
>::type
sum(vector_set_expression<S, Device> const& set) {
	typedef typename device_traits<Device>:: template add<typename S::value_type> Add;
	typedef typename device_traits<Device>:: template identity<typename S::value_type> Identity;
	return detail::fold_vector_set_optimizer<S, Add, Identity>::create(set(), Add(), Identity());
}

/// \brief Computes the maximum of elements for each point in the set
///
/// Formula: (max S)_i = max_j S_ij
template<class S, class Device>
typename detail::fold_vector_set_optimizer<
	S, typename device_traits<Device>:: template max<typename S::value_type>
	, typename device_traits<Device>:: template identity<typename S::value_type>
>::type
max(vector_set_expression<S, Device> const& set) {
	typedef typename device_traits<Device>:: template max<typename S::value_type> Max;
	typedef typename device_traits<Device>:: template identity<typename S::value_type> Identity;
	return detail::fold_vector_set_optimizer<S, Max, Identity>::create(set(), Max(), Identity());
}

/// \brief Computes the minimum of elements for each point in the set
///
/// Formula: (min S)_i = min_j S_ij
template<class S, class Device>
typename detail::fold_vector_set_optimizer<
	S, typename device_traits<Device>:: template min<typename S::value_type>
	, typename device_traits<Device>:: template identity<typename S::value_type>
>::type
min(vector_set_expression<S, Device> const& set) {
	typedef typename device_traits<Device>:: template min<typename S::value_type> Min;
	typedef typename device_traits<Device>:: template identity<typename S::value_type> Identity;
	return detail::fold_vector_set_optimizer<S, Min, Identity>::create(set(), Min(), Identity());
}

//~ /// \brief arg_max v = arg max_i v_i
//~ template<class M, class Device>
//~ std::size_t arg_max(vector_set_expression<M, Device> const& set) {
	//~ return kernels::vector_max(elem_result);
//~ }

//~ /// \brief arg_min v = arg min_i v_i
//~ template<class VecV, class Device>
//~ auto arg_min(vector_expression<VecV, Device> const& v) -> decltype(arg_max(-v)){
	//~ return arg_max(-v);
//~ }

/// \brief Computes norm_1 for each element in the set.
///
/// Formula: norm_1(S)_i = norm_1(point(S,i)) 
template<class S, class Device>
auto norm_1(vector_set_expression<S, Device> const& set) ->decltype(sum(abs(set))) {
	return sum(abs(set));
}

/// \brief Computes norm_sqr for each element in the set.
///
/// Formula: norm_sqr(S)_i = norm_sqr(point(S,i)) 
template<class S, class Device>
auto norm_sqr(vector_set_expression<S, Device> const& set) ->decltype(sum(sqr(set))){
	return sum(sqr(set));
}

/// \brief Computes norm_2 for each element in the set.
///
/// Formula: norm_2(S)_i = norm_2(point(S,i)) 
template<class S, class Device>
auto norm_2(vector_set_expression<S, Device> const& set) ->decltype(sqrt(norm_sqr(set))){
	return sqrt(norm_sqr(set));
}

/// \brief Computes norm_inf for each element in the set.
///
/// Formula: norm_inf(S)_i = norm_inf(point(S,i)) 
template<class S, class Device>
auto norm_inf(vector_set_expression<S, Device> const& set) ->decltype(max(abs(set))){
	return max(abs(set));
}


/////////////////////////////////////////
//////////MATRIX REDUCTIONS
/////////////////////////////////////////

/// \brief Computes the elementwise sum over all elements of A
///
/// returns a scalar s = sum_ij A_ij
template<class MatA, class Device>
typename MatA::value_type sum(matrix_expression<MatA, Device> const& A){
	typedef typename MatA::value_type value_type;
	typedef typename device_traits<Device>:: template add<value_type> functor_type;
	auto const& elem_result = eval_block(A);
	value_type result = 0;
	kernels::matrix_fold<functor_type>(elem_result,result);
	return result;
}



/// \brief Computes the elementwise maximum over all elements of A
///
/// returns a scalar s = max_ij A_ij
template<class MatA, class Device>
typename MatA::value_type max(matrix_expression<MatA, Device> const& A){
	typedef typename MatA::value_type value_type;
	typedef typename device_traits<Device>:: template max<value_type> functor_type;
	auto const& elem_result = eval_block(A);
	value_type result = 0;
	kernels::matrix_fold<functor_type>(elem_result,result);
	return result;
}

/// \brief Computes the elementwise minimum over all elements of A
///
/// returns a scalar s = min_ij A_ij
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
){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	return sum(eval_block(A*B));
}

/// \brief Computes the matrix 1-norm |A|_1
/// 
/// It is defined as \f$ \max_i \sum_j |A_{ij}| \f$ 
template<class MatA, class Device>
typename real_traits<typename MatA::value_type>::type
norm_1(matrix_expression<MatA, Device> const& A) {
	return max(norm_1(as_columns(A)));
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
	return max(norm_1(as_rows(A)));
}

/// \brief Evaluates the trace of matrix A
///
/// The trace is defined as the sum of the diagonal elements of A,
/// \f$ \text{trace}(A) = \sum_i A_{ii}\f$
///
/// \param  A square matrix
/// \return the sum of the values at the diagonal of \em A
template < class MatA, class Device>
typename MatA::value_type trace(matrix_expression<MatA, Device> const& A){
	REMORA_SIZE_CHECK(A().size1() == A().size2());
	return sum(diag(A));
}

/////////////////////////////////////////
//////// Block-Matrix Creation
/////////////////////////////////////////

/// \brief Forms the block matrix (A|B) where B is to the right of A
template<class MatA, class MatB, class Device>
matrix_concat<MatA, MatB, true> operator|(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	return matrix_concat<MatA, MatB, true>(A(),B());
}

/// \brief Forms the block matrix (A|v) where v is a column vector to the right of A
template<class MatA, class VecV, class Device>
auto operator|(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device> const& v
) -> decltype(A | trans(repeat(v,1))){
	return A | trans(repeat(v,1));
}

/// \brief Forms the block matrix (v|A) where v is a column vector to the left of A
template<class MatA, class VecV, class Device>
auto operator|(
	vector_expression<VecV, Device> const& v,
	matrix_expression<MatA, Device> const& A
) -> decltype(trans(repeat(v,1)) | A){
	return trans(repeat(v,1)) | A;
}

/// \brief Forms the block matrix (A|t)
///
/// The scalar t is interpreted as column vector
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value, 
	matrix_concat<MatA, scalar_matrix<T, Device, row_major>, true > 
>::type operator|(
	matrix_expression<MatA, Device> const& A,
	T const& t
){
	return A | repeat(t,A().size1(), 1);
}

/// \brief Forms the block matrix (t|A)
///
/// The scalar t is interpreted as column vector
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value, 
	matrix_concat<scalar_matrix<T, Device, row_major>, MatA, true > 
>::type operator|(
	T const& t,
	matrix_expression<MatA, Device> const& A
){
	return repeat(t,A().size1(), 1) | A;
}

///\brief Forms the block matrix A&B where A is on top of B
template<class MatA, class MatB, class Device>
matrix_concat<MatA, MatB, false> operator&(
	matrix_expression<MatA, Device> const& A,
	matrix_expression<MatB, Device> const& B
){
	return matrix_concat<MatA, MatB, false>(A(),B());
}

/// \brief Forms the block matrix (A & v) where v is a row vector on the bottom of A
template<class MatA, class VecV, class Device>
auto operator&(
	matrix_expression<MatA, Device> const& A,
	vector_expression<VecV, Device> const& v
) -> decltype(A & repeat(v,1)){
	return A & repeat(v,1);
}

/// \brief Forms the block matrix (A & v) where v is a row vector on the top of A
template<class MatA, class VecV, class Device>
auto operator&(
	vector_expression<VecV, Device> const& v,
	matrix_expression<MatA, Device> const& A
) -> decltype(repeat(v,1) & A){
	return repeat(v,1) & A;
}

/// \brief Forms the block matrix (A & t)
///
/// The scalar t is interpreted as row vector
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value, 
	matrix_concat<MatA, scalar_matrix<T, Device, row_major>, false > 
>::type operator&(
	matrix_expression<MatA, Device> const& A,
	T const& t
){
	return A & repeat(t, 1, A().size2());
}

/// \brief Forms the block matrix (t & A)
///
/// The scalar t is interpreted as row vector
template<class MatA, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename MatA::value_type>::value, 
	matrix_concat<scalar_matrix<T, Device, row_major>, MatA, false > 
>::type operator&(
	T const& t,
	matrix_expression<MatA, Device> const& A
){
	return repeat(t,1, A().size2()) & A;
}

}

#endif
