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
#ifndef REMORA_VECTOR_EXPRESSION_HPP
#define REMORA_VECTOR_EXPRESSION_HPP

#include "detail/expression_optimizers.hpp"
#include "kernels/dot.hpp"
#include "kernels/vector_fold.hpp"
#include "kernels/vector_max.hpp"

namespace remora{

template<class T, class VecV, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type >::value,
	typename detail::vector_scalar_multiply_optimizer<VecV>::type
>::type
operator* (vector_expression<VecV, Device> const& v, T scalar){
	typedef typename VecV::value_type value_type;
	return detail::vector_scalar_multiply_optimizer<VecV>::create(v(),value_type(scalar));
}
template<class T, class VecV, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type >::value,
        typename detail::vector_scalar_multiply_optimizer<VecV>::type
>::type
operator* (T scalar, vector_expression<VecV, Device> const& v){
	typedef typename VecV::value_type value_type;
	return detail::vector_scalar_multiply_optimizer<VecV>::create(v(),value_type(scalar));
}

template<class VecV, class Device>
typename detail::vector_scalar_multiply_optimizer<VecV>::type
operator-(vector_expression<VecV, Device> const& v){
	typedef typename VecV::value_type value_type;
	return detail::vector_scalar_multiply_optimizer<VecV>::create(v(),value_type(-1));
}

///\brief Creates a vector having a constant value.
///
///@param scalar the value which is repeated
///@param elements the size of the resulting vector
template<class T>
typename std::enable_if<std::is_arithmetic<T>::value, scalar_vector<T, cpu_tag> >::type
repeat(T scalar, std::size_t elements){
	return scalar_vector<T, cpu_tag>(elements,scalar);
}


#define REMORA_UNARY_VECTOR_TRANSFORMATION(name, F)\
template<class VecV, class Device>\
typename detail::vector_unary_optimizer<VecV,typename device_traits<Device>:: template F<typename VecV::value_type> >::type \
name(vector_expression<VecV, Device> const& v){\
	typedef typename device_traits<Device>:: template F<typename VecV::value_type> functor_type;\
	return detail::vector_unary_optimizer<VecV, functor_type >::create(v(), functor_type());\
}
REMORA_UNARY_VECTOR_TRANSFORMATION(abs, abs)
REMORA_UNARY_VECTOR_TRANSFORMATION(log, log)
REMORA_UNARY_VECTOR_TRANSFORMATION(exp, exp)
REMORA_UNARY_VECTOR_TRANSFORMATION(tanh,tanh)
REMORA_UNARY_VECTOR_TRANSFORMATION(sin,sin)
REMORA_UNARY_VECTOR_TRANSFORMATION(cos,cos)
REMORA_UNARY_VECTOR_TRANSFORMATION(tan,tan)
REMORA_UNARY_VECTOR_TRANSFORMATION(asin,asin)
REMORA_UNARY_VECTOR_TRANSFORMATION(acos,acos)
REMORA_UNARY_VECTOR_TRANSFORMATION(atan,atan)
REMORA_UNARY_VECTOR_TRANSFORMATION(erf,erf)
REMORA_UNARY_VECTOR_TRANSFORMATION(erfc,erfc)
REMORA_UNARY_VECTOR_TRANSFORMATION(sqr, sqr)
REMORA_UNARY_VECTOR_TRANSFORMATION(sqrt, sqrt)
REMORA_UNARY_VECTOR_TRANSFORMATION(cbrt, cbrt)
REMORA_UNARY_VECTOR_TRANSFORMATION(sigmoid, sigmoid)
REMORA_UNARY_VECTOR_TRANSFORMATION(softPlus, soft_plus)
REMORA_UNARY_VECTOR_TRANSFORMATION(elem_inv, inv)
#undef REMORA_UNARY_VECTOR_TRANSFORMATION

///\brief Adds two vectors
template<class VecV1, class VecV2, class Device>
vector_addition<VecV1, VecV2 > operator+ (
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
){
	REMORA_SIZE_CHECK(v1().size() == v2().size());
	return vector_addition<VecV1, VecV2>(v1(),v2());
}
///\brief Subtracts two vectors
template<class VecV1, class VecV2, class Device>
auto operator- (
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
) -> decltype (v1 + (-v2)){
	REMORA_SIZE_CHECK(v1().size() == v2().size());
	return v1 + (-v2);
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class VecV, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value, 
	vector_addition<VecV, scalar_vector<T, Device> >
>::type operator+ (
	vector_expression<VecV, Device> const& v,
	T t
){
	return v + scalar_vector<T, Device>(v().size(),t);
}

///\brief Adds a vector plus a scalar which is interpreted as a constant vector
template<class T, class VecV, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value,
	vector_addition<VecV, scalar_vector<T, Device> >
>::type operator+ (
	T t,
	vector_expression<VecV, Device> const& v
){
	return v + scalar_vector<T, Device>(v().size(),t);
}

///\brief Subtracts a scalar which is interpreted as a constant vector.
template<class VecV, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value ,
	decltype(std::declval<VecV const&>() + T())
>::type operator- (
	vector_expression<VecV, Device> const& v,
	T t
){
	return v + (-t);
}

///\brief Subtracts a vector from a scalar which is interpreted as a constant vector
template<class VecV, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value,
	decltype(T() + (-std::declval<VecV const&>()))
>::type operator- (
	T t,
	vector_expression<VecV, Device> const& v
){
	return t + (-v);
}

#define REMORA_BINARY_VECTOR_EXPRESSION(name, F)\
template<class VecV1, class VecV2, class Device>\
vector_binary<VecV1, VecV2, typename device_traits<Device>:: template F<typename common_value_type<VecV1,VecV2>::type> >\
name(vector_expression<VecV1, Device> const& v1, vector_expression<VecV2, Device> const& v2){\
	REMORA_SIZE_CHECK(v1().size() == v2().size());\
	typedef typename common_value_type<VecV1,VecV2>::type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return vector_binary<VecV1, VecV2, functor_type >(v1(),v2(), functor_type());\
}
REMORA_BINARY_VECTOR_EXPRESSION(operator*, multiply)
REMORA_BINARY_VECTOR_EXPRESSION(element_prod, multiply)
REMORA_BINARY_VECTOR_EXPRESSION(operator/, divide)
REMORA_BINARY_VECTOR_EXPRESSION(element_div, divide)
REMORA_BINARY_VECTOR_EXPRESSION(pow, pow)
REMORA_BINARY_VECTOR_EXPRESSION(min, min)
REMORA_BINARY_VECTOR_EXPRESSION(max, max)
#undef REMORA_BINARY_VECTOR_EXPRESSION


//operations of the form op(v,t)[i] = op(v[i],t)
#define REMORA_VECTOR_SCALAR_TRANSFORMATION(name, F)\
template<class T, class VecV, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename VecV::value_type >::value,\
        vector_binary<VecV, scalar_vector<typename VecV::value_type, Device>, \
	typename device_traits<Device>:: template F<typename VecV::value_type> > \
>::type \
name (vector_expression<VecV, Device> const& v, T t){ \
	typedef typename VecV::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return  vector_binary<VecV, scalar_vector<type, Device>, functor_type >(v(), scalar_vector<type, Device>(v().size(),(type)t), functor_type()); \
}
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator/, divide)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator<, less)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator<=, less_equal)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator>, greater)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator>=, greater_equal)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator==, equal)
REMORA_VECTOR_SCALAR_TRANSFORMATION(operator!=, not_equal)
REMORA_VECTOR_SCALAR_TRANSFORMATION(min, min)
REMORA_VECTOR_SCALAR_TRANSFORMATION(max, max)
REMORA_VECTOR_SCALAR_TRANSFORMATION(pow, pow)
#undef REMORA_VECTOR_SCALAR_TRANSFORMATION

// operations of the form op(t,v)[i] = op(t,v[i])
#define REMORA_VECTOR_SCALAR_TRANSFORMATION_2(name, F)\
template<class T, class VecV, class Device> \
typename std::enable_if< \
	std::is_convertible<T, typename VecV::value_type >::value,\
	vector_binary<scalar_vector<typename VecV::value_type, Device>, VecV, typename device_traits<Device>:: template F<typename VecV::value_type> > \
>::type \
name (T t, vector_expression<VecV, Device> const& v){ \
	typedef typename VecV::value_type type;\
	typedef typename device_traits<Device>:: template F<type> functor_type;\
	return  vector_binary<scalar_vector<T, Device>, VecV, functor_type >(scalar_vector<T, Device>(v().size(),t), v() ,functor_type()); \
}
REMORA_VECTOR_SCALAR_TRANSFORMATION_2(min, min)
REMORA_VECTOR_SCALAR_TRANSFORMATION_2(max, max)
#undef REMORA_VECTOR_SCALAR_TRANSFORMATION_2

template<class VecV1, class VecV2, class Device>
vector_binary<VecV1, VecV2, 
	typename device_traits<Device>:: template safe_divide<typename common_value_type<VecV1,VecV2>::type > 
>
safe_div(
	vector_expression<VecV1, Device> const& v1, 
	vector_expression<VecV2, Device> const& v2, 
	typename common_value_type<VecV1,VecV2>::type defaultValue
){
	REMORA_SIZE_CHECK(v1().size() == v2().size());
	typedef typename common_value_type<VecV1,VecV2>::type result_type;
	
	typedef typename device_traits<Device>:: template safe_divide<result_type> functor_type;
	return vector_binary<VecV1, VecV2, functor_type>(v1(),v2(), functor_type(defaultValue));
}

/////VECTOR REDUCTIONS

/// \brief sum v = sum_i v_i
template<class VecV, class Device>
typename VecV::value_type
sum(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	typedef typename device_traits<Device>:: template add<value_type> functor_type;
	auto const& elem_result = eval_block(v);
	value_type result = 0;
	kernels::vector_fold<functor_type>(elem_result,result);
	return result;
}

/// \brief max v = max_i v_i
template<class VecV, class Device>
typename VecV::value_type
max(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	typedef typename device_traits<Device>:: template max<value_type> functor_type;
	auto const& elem_result = eval_block(v);
	value_type result = std::numeric_limits<value_type>::lowest();
	kernels::vector_fold<functor_type>(elem_result,result);
	return result;
}

/// \brief min v = min_i v_i
template<class VecV, class Device>
typename VecV::value_type
min(vector_expression<VecV, Device> const& v) {
	typedef typename VecV::value_type value_type;
	typedef typename device_traits<Device>:: template min<value_type> functor_type;
	auto const& elem_result = eval_block(v);
	value_type result = std::numeric_limits<value_type>::max();
	kernels::vector_fold<functor_type>(elem_result,result);
	return result;
}

/// \brief arg_max v = arg max_i v_i
template<class VecV, class Device>
std::size_t arg_max(vector_expression<VecV, Device> const& v) {
	REMORA_SIZE_CHECK(v().size() > 0);
	auto const& elem_result = eval_block(v);
	return kernels::vector_max(elem_result);
}

/// \brief arg_min v = arg min_i v_i
template<class VecV, class Device>
std::size_t arg_min(vector_expression<VecV, Device> const& v) {
	REMORA_SIZE_CHECK(v().size() > 0);
	return arg_max(-v);
}

/// \brief soft_max v = ln(sum(exp(v)))
///
/// Be aware that this is NOT the same function as used in machine learning: exp(v)/sum(exp(v))
///
/// The function is computed in an numerically stable way to prevent that too high values of v_i produce inf or nan.
/// The name of the function comes from the fact that it behaves like a continuous version of max in the respect that soft_max v <= v.size()*max(v)
/// max is reached in the limit as the gap between the biggest value and the rest grows to infinity.
template<class VecV, class Device>
typename VecV::value_type
soft_max(vector_expression<VecV, Device> const& v) {
	typename VecV::value_type maximum = max(v);
	using std::log;
	return log(sum(exp(v - maximum))) + maximum;
}


////implement all the norms based on sum!

/// \brief norm_1 v = sum_i |v_i|
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_1(vector_expression<VecV, Device> const& v) {
	return sum(abs(eval_block(v)));
}

/// \brief norm_2 v = sum_i |v_i|^2
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_sqr(vector_expression<VecV, Device> const& v) {
	return sum(sqr(eval_block(v)));
}

/// \brief norm_2 v = sqrt (sum_i |v_i|^2 )
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_2(vector_expression<VecV, Device> const& v) {
	using std::sqrt;
	return sqrt(norm_sqr(v));
}

/// \brief norm_inf v = max_i |v_i|
template<class VecV, class Device>
typename real_traits<typename VecV::value_type >::type
norm_inf(vector_expression<VecV, Device> const& v){
	return max(abs(eval_block(v)));
}

/// \brief index_norm_inf v = arg max_i |v_i|
template<class VecV, class Device>
std::size_t index_norm_inf(vector_expression<VecV, Device> const& v){
	return arg_max(abs(eval_block(v)));
}

// inner_prod (v1, v2) = sum_i v1_i * v2_i
template<class VecV1, class VecV2, class Device>
decltype(
	typename VecV1::value_type() * typename VecV2::value_type()
)
inner_prod(
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
) {
	REMORA_SIZE_CHECK(v1().size() == v2().size());
	typedef decltype(
		typename VecV1::value_type() * typename VecV2::value_type()
	) value_type;
	value_type result = value_type();
	kernels::dot(eval_block(v1),eval_block(v2),result);
	return result;
}


// Vector Concatenation


///\brief Concatenates two vectors
///
/// Given two vectors v and w, forms the vector (v,w). 
template<class VecV1, class VecV2, class Device>
vector_concat<VecV1,VecV2> operator|(
	vector_expression<VecV1, Device> const& v1,
	vector_expression<VecV2, Device> const& v2
){
	return vector_concat<VecV1,VecV2>(v1(),v2());
}

///\brief Concatenates a vector with a scalar
///
/// Given a vector v and a scalar t, forms the vector (v,t)
template<class VecV, class T, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value, 
	vector_concat<VecV, scalar_vector<T, Device> >
>::type operator| (
	vector_expression<VecV, Device> const& v,
	T t
){
	return v | scalar_vector<T, Device>(1,t);
}

///\brief Concatenates a vector with a scalar
///
/// Given a vector v and a scalar t, forms the vector (v,t)
template<class T, class VecV, class Device>
typename std::enable_if<
	std::is_convertible<T, typename VecV::value_type>::value,
	vector_concat<scalar_vector<T, Device>,VecV >
>::type operator| (
	T t,
	vector_expression<VecV, Device> const& v
){
	return scalar_vector<T, Device>(1,t) | v;
}


}

#endif
