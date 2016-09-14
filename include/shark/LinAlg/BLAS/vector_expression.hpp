/*!
 * \brief       expression templates for vector valued math
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
#ifndef SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP
#define SHARK_LINALG_BLAS_VECTOR_EXPRESSION_HPP

#include "detail/expression_optimizers.hpp"
#include "kernels/dot.hpp"
#include <boost/utility/enable_if.hpp>

namespace shark {
namespace blas {

template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        vector_scalar_multiply<E>
>::type
operator* (vector_expression<E> const& e, T scalar){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e(), scalar_type(scalar));
}
template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::scalar_type >,
        vector_scalar_multiply<E>
>::type
operator* (T scalar, vector_expression<E> const& e){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e(), scalar_type(scalar));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}

template<class E>
vector_scalar_multiply<E> operator-(vector_expression<E> const& e){
	typedef typename E::scalar_type scalar_type;
	return vector_scalar_multiply<E>(e(), scalar_type(-1));//explicit cast prevents warning, alternative would be to template vector_scalar_multiply on T as well
}

///\brief Creates a vector having a constant value.
///
///@param scalar the value which is repeated
///@param elements the size of the resulting vector
template<class T>
typename boost::enable_if<std::is_arithmetic<T>, scalar_vector<T> >::type
repeat(T scalar, std::size_t elements){
	return scalar_vector<T>(elements,scalar);
}


#define SHARK_UNARY_VECTOR_TRANSFORMATION(name, F)\
template<class E>\
vector_unary<E,F<typename E::value_type> >\
name(vector_expression<E> const& e){\
	typedef F<typename E::value_type> functor_type;\
	return vector_unary<E, functor_type>(e(), functor_type());\
}
SHARK_UNARY_VECTOR_TRANSFORMATION(abs, scalar_abs)
SHARK_UNARY_VECTOR_TRANSFORMATION(log, scalar_log)
SHARK_UNARY_VECTOR_TRANSFORMATION(exp, scalar_exp)
SHARK_UNARY_VECTOR_TRANSFORMATION(cos, scalar_cos)
SHARK_UNARY_VECTOR_TRANSFORMATION(sin, scalar_sin)
SHARK_UNARY_VECTOR_TRANSFORMATION(tanh,scalar_tanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(atanh,scalar_atanh)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqr, scalar_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(abs_sqr, scalar_abs_sqr)
SHARK_UNARY_VECTOR_TRANSFORMATION(sqrt, scalar_sqrt)
SHARK_UNARY_VECTOR_TRANSFORMATION(sigmoid, scalar_sigmoid)
SHARK_UNARY_VECTOR_TRANSFORMATION(softPlus, scalar_soft_plus)
SHARK_UNARY_VECTOR_TRANSFORMATION(elem_inv, scalar_inverse)
#undef SHARK_UNARY_VECTOR_TRANSFORMATION


//operations of the form op(v,t)[i] = op(v[i],t)
#define SHARK_VECTOR_SCALAR_TRANSFORMATION(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	std::is_convertible<T, typename E::value_type >,\
        vector_unary<E,F<typename E::value_type,T> > \
>::type \
name (vector_expression<E> const& e, T scalar){ \
	typedef F<typename E::value_type,T> functor_type; \
	return vector_unary<E, functor_type>(e(), functor_type(scalar)); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator/, scalar_divide)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<, scalar_less_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator<=, scalar_less_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>, scalar_bigger_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator>=, scalar_bigger_equal_than)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator==, scalar_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(operator!=, scalar_not_equal)
SHARK_VECTOR_SCALAR_TRANSFORMATION(min, scalar_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION(max, scalar_max)
SHARK_VECTOR_SCALAR_TRANSFORMATION(pow, scalar_pow)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i] = op(t,v[i])
#define SHARK_VECTOR_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class E> \
typename boost::enable_if< \
	std::is_convertible<T, typename E::value_type >,\
        vector_unary<E,F<typename E::value_type,T> > \
>::type \
name (T scalar, vector_expression<E> const& e){ \
	typedef F<typename E::value_type,T> functor_type; \
	return vector_unary<E, functor_type>(e(), functor_type(scalar)); \
}
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(min, scalar_min)
SHARK_VECTOR_SCALAR_TRANSFORMATION_2(max, scalar_max)
#undef SHARK_VECTOR_SCALAR_TRANSFORMATION_2



///\brief Adds two vectors
template<class E1, class E2>
vector_addition<E1, E2 > operator+ (
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
){
	return vector_addition<E1, E2>(e1(),e2());
}
///\brief Subtracts two vectors
template<class E1, class E2>
vector_addition<E1, vector_scalar_multiply<E2> > operator- (
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
){
	return vector_addition<E1, vector_scalar_multiply<E2> >(e1(),-e2());
}

///\brief Adds a vector plus a scalr which is interpreted as a constant vector
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>, 
	vector_addition<E, scalar_vector<T> >
>::type operator+ (
	vector_expression<E> const& e,
	T t
){
	return e + scalar_vector<T>(e().size(),t);
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class T, class E>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>,
	vector_addition<E, scalar_vector<T> >
>::type operator+ (
	T t,
	vector_expression<E> const& e
){
	return e + scalar_vector<T>(e().size(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant vector from a vector.
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type> ,
	vector_addition<E, vector_scalar_multiply<scalar_vector<T> > >
>::type operator- (
	vector_expression<E> const& e,
	T t
){
	return e - scalar_vector<T>(e().size(),t);
}

///\brief Subtracts a vector from a scalar which is interpreted as a constant vector
template<class E, class T>
typename boost::enable_if<
	std::is_convertible<T, typename E::value_type>,
	vector_addition<scalar_vector<T>, vector_scalar_multiply<E> >
>::type operator- (
	T t,
	vector_expression<E> const& e
){
	return scalar_vector<T>(e().size(),t) - e;
}




#define SHARK_BINARY_VECTOR_EXPRESSION(name, F)\
template<class E1, class E2>\
vector_binary<E1, E2, F<typename E1::value_type, typename E2::value_type> >\
name(vector_expression<E1> const& e1, vector_expression<E2> const& e2){\
	typedef F<typename E1::value_type, typename E2::value_type> functor_type;\
	return vector_binary<E1, E2, functor_type>(e1(),e2(), functor_type());\
}
SHARK_BINARY_VECTOR_EXPRESSION(operator*, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(element_prod, scalar_binary_multiply)
SHARK_BINARY_VECTOR_EXPRESSION(operator/, scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(element_div, scalar_binary_divide)
SHARK_BINARY_VECTOR_EXPRESSION(min, scalar_binary_min)
SHARK_BINARY_VECTOR_EXPRESSION(max, scalar_binary_max)
#undef SHARK_BINARY_VECTOR_EXPRESSION

template<class E1, class E2>
vector_binary<E1, E2, 
	scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> 
>
safe_div(
	vector_expression<E1> const& e1, 
	vector_expression<E2> const& e2, 
	decltype(
		typename E1::value_type() * typename E2::value_type()
	) defaultValue
){
	typedef scalar_binary_safe_divide<typename E1::value_type, typename E2::value_type> functor_type;
	return vector_binary<E1, E2, functor_type>(e1(),e2(), functor_type(defaultValue));
}

/////VECTOR REDUCTIONS

/// \brief sum v = sum_i v_i
template<class E>
typename E::value_type
sum(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_plus<value_type, value_type> > kernel;
	return kernel(eval_block(e),value_type());
}

/// \brief max v = max_i v_i
template<class E>
typename E::value_type
max(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_max<value_type, value_type> > kernel;
	auto const& elem_result = eval_block(e);
	return kernel(elem_result,elem_result(0));
}

/// \brief min v = min_i v_i
template<class E>
typename E::value_type
min(const vector_expression<E> &e) {
	typedef typename E::value_type value_type;
	vector_fold<scalar_binary_min<value_type, value_type> > kernel;
	auto const& elem_result = eval_block(e);
	return kernel(elem_result,elem_result(0));
}

/// \brief arg_max v = arg max_i v_i
template<class E>
std::size_t arg_max(const vector_expression<E> &e) {
	SIZE_CHECK(e().size() > 0);
	auto const& elem_result = eval_block(e);
	return std::max_element(elem_result.begin(),elem_result.end()).index();
}

/// \brief arg_min v = arg min_i v_i
template<class E>
std::size_t arg_min(const vector_expression<E> &e) {
	SIZE_CHECK(e().size() > 0);
	return arg_max(-e);
}

/// \brief soft_max v = ln(sum(exp(v)))
///
/// Be aware that this is NOT the same function as used in machine learning: exp(v)/sum(exp(v))
///
/// The function is computed in an numerically stable way to prevent that too high values of v_i produce inf or nan.
/// The name of the function comes from the fact that it behaves like a continuous version of max in the respect that soft_max v <= v.size()*max(v)
/// max is reached in the limit as the gap between the biggest value and the rest grows to infinity.
template<class E>
typename E::value_type
soft_max(const vector_expression<E> &e) {
	typename E::value_type maximum = max(e);
	return std::log(sum(exp(e - maximum))) + maximum;
}


////implement all the norms based on sum!

/// \brief norm_1 v = sum_i |v_i|
template<class E>
typename real_traits<typename E::value_type >::type
norm_1(const vector_expression<E> &e) {
	return sum(abs(eval_block(e)));
}

/// \brief norm_2 v = sum_i |v_i|^2
template<class E>
typename real_traits<typename E::value_type >::type
norm_sqr(const vector_expression<E> &e) {
	return sum(abs_sqr(eval_block(e)));
}

/// \brief norm_2 v = sqrt (sum_i |v_i|^2 )
template<class E>
typename real_traits<typename E::value_type >::type
norm_2(const vector_expression<E> &e) {
	using std::sqrt;
	return sqrt(norm_sqr(e));
}

/// \brief norm_inf v = max_i |v_i|
template<class E>
typename real_traits<typename E::value_type >::type
norm_inf(vector_expression<E> const &e){
	return max(abs(eval_block(e)));
}

/// \brief index_norm_inf v = arg max_i |v_i|
template<class E>
std::size_t index_norm_inf(vector_expression<E> const &e){
	return arg_max(abs(eval_block(e)));
}

// inner_prod (v1, v2) = sum_i v1_i * v2_i
template<class E1, class E2>
decltype(
	typename E1::value_type() * typename E2::value_type()
)
inner_prod(
	vector_expression<E1> const& e1,
	vector_expression<E2> const& e2
) {
	typedef decltype(
		typename E1::value_type() * typename E2::value_type()
	) value_type;
	value_type result = value_type();
	kernels::dot(eval_block(e1),eval_block(e2),result);
	return result;
}

}

}

#endif
