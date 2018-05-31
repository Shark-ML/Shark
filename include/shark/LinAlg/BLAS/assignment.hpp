/*!
 * 
 *
 * \brief      Assignment and evaluation of vector expressions
 * 
 * 
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

#ifndef REMORA_ASSIGNMENT_HPP
#define REMORA_ASSIGNMENT_HPP

#include "kernels/matrix_assign.hpp"
#include "kernels/vector_assign.hpp"
#include "detail/traits.hpp"
#include "detail/proxy_optimizers_fwd.hpp"

#include <type_traits>
namespace remora{
	
//////////////////////////////////////////////////////////////////////
/////Evaluate blockwise expressions
//////////////////////////////////////////////////////////////////////

///\brief conditionally evaluates a vector expression if it is a block expression
///
/// If the expression is a block expression, a temporary vector is created to which
/// the expression is assigned, which is then returned, otherwise the expression itself
/// is returned
template<class E, class Device>
typename std::conditional<
	std::is_base_of<
		blockwise_tag,
		typename E::evaluation_category
	>::value,
	typename vector_temporary<E>::type,
	E const&
>::type
eval_block(vector_expression<E, Device> const& e){
	return e();//either casts to E const& or returns the copied expression
}
///\brief conditionally evaluates a matrix expression if it is a block expression
///
/// If the expression is a block expression, a temporary matrix is created to which
/// the expression is assigned, which is then returned, otherwise the expression itself
/// is returned
template<class E, class Device>
typename std::conditional<
	std::is_base_of<
		blockwise_tag,
		typename E::evaluation_category
	>::value,
	typename matrix_temporary<E>::type,
	E const&
>::type
eval_block(matrix_expression<E, Device> const& e){
	return e();//either casts to E const& or returns the copied expression
}


///\brief Evaluates an expression if it does not have a standard storage layout
///
/// This function evaluates an expression to a temporary if it does not have
/// a known storage type. i.e. proxy expressions and containers are not evaluated but passed
/// through while everything else is evaluated.
template<class E, class Device>
typename std::conditional<
	std::is_same<
		unknown_storage,
		typename E::storage_type
	>::value,
	typename vector_temporary<E>::type,
	E const&
>::type
eval_expression(vector_expression<E, Device> const& e){
	return e();//either casts to E const& or returns the evaluated expression
}
///\brief Evaluates an expression if it does not have a standard storage layout
///
/// This function evaluates an expression to a temporary if it does not have
/// a known storage type. i.e. proxy expressions and containers are not evaluated but passed
/// through while everything else is evaluated.
template<class E, class Device>
typename std::conditional<
	std::is_same<
		unknown_storage,
		typename E::storage_type
	>::value,
	typename matrix_temporary<E>::type,
	E const&
>::type
eval_expression(matrix_expression<E, Device> const& e){
	return e();//either casts to E const& or returns the evaluated expression
}
	
/////////////////////////////////////////////////////////////////////////////////////
////// Vector Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class VecX, class VecV, class Device>
	void assign(
		vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v,
		elementwise_tag
	){
		kernels::assign(x, v);
	}
	template<class VecX, class VecV, class Device>
	void assign(
		vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v,
		blockwise_tag
	){
		v().assign_to(x);
	}
	template<class VecX, class VecV, class Device>
	void plus_assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v,
		elementwise_tag
	){
		typename device_traits<Device>:: template add<typename common_value_type<VecX,VecV>::type> f;
		kernels::assign(x, v, f);
	}
	template<class VecX, class VecV, class Device>
	void plus_assign(
		vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v,
		blockwise_tag
	){
		v().plus_assign_to(x);
	}
}

/// \brief Dispatches vector assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV, class Device>
VecX& assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	detail::assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV, class Device>
VecX& plus_assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	detail::plus_assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector minus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV, class Device>
VecX& minus_assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	
	typedef typename VecV::value_type value_type;
	auto minusV = detail::vector_scalar_multiply_optimizer<VecV>::create(v(),value_type(-1));
	detail::plus_assign(x,minusV,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV, class Device>
VecX& multiply_assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	auto&& veval = eval_block(v);
	typedef typename device_traits<Device>:: template multiply<typename common_value_type<VecX,VecV>::type> F;
	kernels::assign(x, veval, F());
	return x();
}

/// \brief Dispatches vector multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV, class Device>
VecX& divide_assign(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	auto&& veval = eval_block(v);
	typedef typename device_traits<Device>:: template divide<typename common_value_type<VecX,VecV>::type> F;
	kernels::assign(x, veval, F());
	return x();
}
	
/////////////////////////////////////////////////////////////////////////////////////
////// Matrix Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class MatA, class MatB, class Device>
	void assign(
		matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B,
		elementwise_tag
	){
		kernels::assign(A, B);
	}
	template<class MatA, class MatB, class Device>
	void assign(
		matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B,
		blockwise_tag
	){
		B().assign_to(A);
	}
	template<class MatA, class MatB, class Device>
	void plus_assign(
		matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B,
		elementwise_tag
	){
		typename device_traits<Device>:: template add<typename common_value_type<MatA,MatB>::type> f;
		kernels::assign(A,B,f);
	}
	template<class MatA, class MatB, class Device>
	void plus_assign(
		matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B,
		blockwise_tag
	){
		B().plus_assign_to(A);
	}
}

/// \brief Dispatches matrix assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB, class Device>
MatA& assign(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	detail::assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB, class Device>
MatA& plus_assign(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	detail::plus_assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix minus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB, class Device>
MatA& minus_assign(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typedef typename MatB::value_type value_type;
	auto minusB = detail::matrix_scalar_multiply_optimizer<MatB>::create(B(),value_type(-1));
	detail::plus_assign(A, minusB, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB, class Device>
MatA& multiply_assign(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	auto&& Beval = eval_block(B);
	typedef typename device_traits<Device>:: template multiply<typename common_value_type<MatA,MatB>::type> F;
	kernels::assign(A, Beval, F());
	return A();
}

/// \brief Dispatches matrix divide-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB, class Device>
MatA& divide_assign(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	auto&& Beval = eval_block(B);
	typedef typename device_traits<Device>:: template divide<typename common_value_type<MatA,MatB>::type> F;
	kernels::assign(A, Beval, F());
	return A();
}

//////////////////////////////////////////////////////////////////////////////////////
///// Vector Operators
/////////////////////////////////////////////////////////////////////////////////////

/// \brief  Add-Assigns two vector expressions
///
/// Performs the operation x_i+=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)+=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator+=(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return plus_assign(x,temporary);
}

template<class VecX, class VecV, class Device>
typename VecX::closure_type operator+=(vector_expression<VecX, Device>&& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return plus_assign(x,temporary);
}

/// \brief  Subtract-Assigns two vector expressions
///
/// Performs the operation x_i-=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)-=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator-=(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return minus_assign(x,temporary);
}

template<class VecX, class VecV, class Device>
typename VecX::closure_type operator-=(vector_expression<VecX, Device>&& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return minus_assign(x,temporary);
}

/// \brief  Multiply-Assigns two vector expressions
///
/// Performs the operation x_i*=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)*=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator*=(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return multiply_assign(x,temporary);
}

template<class VecX, class VecV, class Device>
typename VecX::closure_type operator*=(vector_expression<VecX, Device>&& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	multiply_assign(x,temporary);
}

/// \brief  Divide-Assigns two vector expressions
///
/// Performs the operation x_i/=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)/=v to avoid this if A and B do not alias
template<class VecX, class VecV, class Device>
VecX& operator/=(vector_expression<VecX, Device>& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return divide_assign(x,temporary);
}

template<class VecX, class VecV, class Device>
typename VecX::closure_type operator/=(vector_expression<VecX, Device>&& x, vector_expression<VecV, Device> const& v){
	REMORA_SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	divide_assign(x,temporary);
}

/// \brief  Adds a scalar to all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,VecX&>::type
operator+=(vector_expression<VecX, Device>& x, T t){
	kernels::assign<typename device_traits<Device>:: template add<typename VecX::value_type> > (x, t);
	return x();
}

template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,typename VecX::closure_type>::type
operator+=(vector_expression<VecX, Device>&& x, T t){
	kernels::assign<typename device_traits<Device>:: template add<typename VecX::value_type> > (x, t);
	return x();
}

/// \brief  Subtracts a scalar from all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,VecX&>::type
operator-=(vector_expression<VecX, Device>& x, T t){
	kernels::assign<typename device_traits<Device>:: template subtract<typename VecX::value_type> > (x, t);
	return x();
}

template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,typename VecX::closure_type>::type
operator-=(vector_expression<VecX, Device>&& x, T t){
	kernels::assign<typename device_traits<Device>:: template subtract<typename VecX::value_type> > (x, t);
	return x();
}

/// \brief  Multiplies a scalar with all elements of the vector
///
/// Performs the operation x_i *= t for all elements.
template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,VecX&>::type
operator*=(vector_expression<VecX, Device>& x, T t){
	kernels::assign<typename device_traits<Device>:: template multiply<typename VecX::value_type> > (x, t);
	return x();
}

template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,typename VecX::closure_type>::type
operator*=(vector_expression<VecX, Device>&& x, T t){
	kernels::assign<typename device_traits<Device>:: template multiply<typename VecX::value_type> > (x, t);
	return x();
}

/// \brief  Divides all elements of the vector by a scalar
///
/// Performs the operation x_i /= t for all elements.
template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,VecX&>::type
operator/=(vector_expression<VecX, Device>& x, T t){
	kernels::assign<typename device_traits<Device>:: template divide<typename VecX::value_type> > (x, t);
	return x();
}

template<class VecX, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename VecX::value_type>::value,typename VecX::closure_type>::type
operator/=(vector_expression<VecX, Device>&& x, T t){
	kernels::assign<typename device_traits<Device>:: template divide<typename VecX::value_type> > (x, t);
	return x();
}



//////////////////////////////////////////////////////////////////////////////////////
///// Matrix Operators
/////////////////////////////////////////////////////////////////////////////////////

/// \brief  Add-Assigns two matrix expressions
///
/// Performs the operation A_ij+=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)+=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator+=(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return plus_assign(A,temporary);
}

template<class MatA, class MatB, class Device>
typename MatA::closure_type operator+=(matrix_expression<MatA, Device>&& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return plus_assign(A,temporary);
}

/// \brief  Subtract-Assigns two matrix expressions
///
/// Performs the operation A_ij-=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)-=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator-=(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return minus_assign(A,temporary);
}

template<class MatA, class MatB, class Device>
typename MatA::closure_type operator-=(matrix_expression<MatA, Device>&& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return minus_assign(A,temporary);
}

/// \brief  Multiply-Assigns two matrix expressions
///
/// Performs the operation A_ij*=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)*=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator*=(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return multiply_assign(A,temporary);
}

template<class MatA, class MatB, class Device>
typename MatA::closure_type operator*=(matrix_expression<MatA, Device>&& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return multiply_assign(A,temporary);
}

/// \brief  Divide-Assigns two matrix expressions
///
/// Performs the operation A_ij/=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)/=B to avoid this if A and B do not alias
template<class MatA, class MatB, class Device>
MatA& operator/=(matrix_expression<MatA, Device>& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return divide_assign(A,temporary);
}

template<class MatA, class MatB, class Device>
typename MatA::closure_type operator/=(matrix_expression<MatA, Device>&& A, matrix_expression<MatB, Device> const& B){
	REMORA_SIZE_CHECK(A().size1() == B().size1());
	REMORA_SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return divide_assign(A,temporary);
}

/// \brief  Adds a scalar to all elements of the matrix
///
/// Performs the operation A_ij += t for all elements.
template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,MatA&>::type
operator+=(matrix_expression<MatA, Device>& A, T t){
	kernels::assign<typename device_traits<Device>:: template add<typename MatA::value_type> > (A, t);
	return A();
}

template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,typename MatA::closure_type>::type
operator+=(matrix_expression<MatA, Device>&& A, T t){
	kernels::assign<typename device_traits<Device>:: template add<typename MatA::value_type> > (A, t);
	return A();
}

/// \brief  Subtracts a scalar from all elements of the matrix
///
/// Performs the operation A_ij -= t for all elements.
template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,MatA&>::type
operator-=(matrix_expression<MatA, Device>& A, T t){
	kernels::assign<typename device_traits<Device>:: template subtract<typename MatA::value_type> > (A, t);
	return A();
}

template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,typename MatA::closure_type>::type
operator-=(matrix_expression<MatA, Device>&& A, T t){
	kernels::assign<typename device_traits<Device>:: template subtract<typename MatA::value_type> > (A, t);
	return A();
}

/// \brief  Multiplies a scalar to all elements of the matrix
///
/// Performs the operation A_ij *= t for all elements.
template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,MatA&>::type
operator*=(matrix_expression<MatA, Device>& A, T t){
	kernels::assign<typename device_traits<Device>:: template multiply<typename MatA::value_type> > (A, t);
	return A();
}

template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,typename MatA::closure_type>::type
operator*=(matrix_expression<MatA, Device>&& A, T t){
	kernels::assign<typename device_traits<Device>:: template multiply<typename MatA::value_type> > (A, t);
	return A();
}

/// \brief  Divides all elements of the matrix by a scalar
///
/// Performs the operation A_ij /= t for all elements.
template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,MatA&>::type
operator/=(matrix_expression<MatA, Device>& A, T t){
	kernels::assign<typename device_traits<Device>:: template divide<typename MatA::value_type> > (A, t);
	return A();
}

template<class MatA, class T, class Device>
typename std::enable_if<std::is_convertible<T, typename MatA::value_type>::value,typename MatA::closure_type>::type
operator/=(matrix_expression<MatA, Device>&& A, T t){
	kernels::assign<typename device_traits<Device>:: template divide<typename MatA::value_type> > (A, t);
	return A();
}

// Assignment proxy.
// Provides temporary free assigment when LHS has no alias on RHS
template<class C>
class noalias_proxy{
public:
	typedef typename C::closure_type closure_type;
	typedef typename C::value_type value_type;

	noalias_proxy(C &lval): m_lval(lval) {}

	noalias_proxy(const noalias_proxy &p):m_lval(p.m_lval) {}

	template <class E>
	closure_type &operator= (const E &e) {
		return assign(m_lval, e);
	}

	template <class E>
	closure_type &operator+= (const E &e) {
		return plus_assign(m_lval, e);
	}

	template <class E>
	closure_type &operator-= (const E &e) {
		return minus_assign(m_lval, e);
	}
	
	template <class E>
	closure_type &operator*= (const E &e) {
		return multiply_assign(m_lval, e);
	}

	template <class E>
	closure_type &operator/= (const E &e) {
		return divide_assign(m_lval, e);
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)+=2;
	closure_type &operator+= (value_type t) {
		return m_lval += t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)-=2;
	closure_type &operator-= (value_type t) {
		return m_lval -= t;
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)*=2;
	closure_type &operator*= (value_type t) {
		return m_lval *= t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)/=2;
	closure_type &operator/= (value_type t) {
		return m_lval /= t;
	}

private:
	closure_type m_lval;
};

// Improve syntax of efficient assignment where no aliases of LHS appear on the RHS
//  noalias(lhs) = rhs_expression
template <class C, class Device>
noalias_proxy<C> noalias(matrix_expression<C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C, class Device>
noalias_proxy<C> noalias(vector_expression<C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C, class Device>
noalias_proxy<C> noalias(vector_expression<C, Device>&& rvalue) {
	return noalias_proxy<C>(rvalue());
}
template <class C, class Device>
noalias_proxy<C> noalias(matrix_expression<C, Device>&& lvalue) {
	return noalias_proxy<C> (lvalue());
}

template <class C, class Device>
noalias_proxy<C> noalias(vector_set_expression<C, Device>& lvalue) {
	return noalias_proxy<C> (lvalue());
}


}
#endif