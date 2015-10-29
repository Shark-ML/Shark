/*!
 * 
 *
 * \brief       Matrix proxy classes.
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

#ifndef SHARK_LINALG_BLAS_ASSIGNMENT_HPP
#define SHARK_LINALG_BLAS_ASSIGNMENT_HPP

#include "kernels/matrix_assign.hpp"
#include "kernels/vector_assign.hpp"
#include "detail/traits.hpp"

namespace shark {
namespace blas {
	
/////////////////////////////////////////////////////////////////////////////////////
////// Vector Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class VecX, class VecV>
	void assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,elementwise_tag){
		kernels::assign(x,v);
	}
	template<class VecX, class VecV>
	void assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,blockwise_tag){
		v().assign_to(x);
	}
	template<class VecX, class VecV>
	void plus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,elementwise_tag){
		kernels::assign<scalar_plus_assign> (x, v);
	}
	template<class VecX, class VecV>
	void plus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,blockwise_tag){
		v().plus_assign_to(x);
	}
	template<class VecX, class VecV>
	void minus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,elementwise_tag){
		kernels::assign<scalar_minus_assign> (x, v);
	}
	template<class VecX, class VecV>
	void minus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,blockwise_tag){
		v().minus_assign_to(x);
	}
	template<class VecX, class VecV>
	void multiply_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,elementwise_tag){
		kernels::assign<scalar_multiply_assign> (x, v);
	}
	template<class VecX, class VecV>
	void multiply_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,blockwise_tag){
		typename vector_temporary<VecX>::type temporary(v);
		kernels::assign<scalar_multiply_assign> (x, temporary);
	}
	template<class VecX, class VecV>
	void divide_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,elementwise_tag){
		kernels::assign<scalar_divide_assign> (x, v);
	}
	template<class VecX, class VecV>
	void divide_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v,blockwise_tag){
		typename vector_temporary<VecX>::type temporary(v);
		kernels::assign<scalar_divide_assign> (x, temporary);
	}
}
	

/// \brief Dispatches vector assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV>
VecX& assign(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	detail::assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV>
VecX& plus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	detail::plus_assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector minus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV>
VecX& minus_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	detail::minus_assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV>
VecX& multiply_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	detail::multiply_assign(x,v,typename VecV::evaluation_category());
	return x();
}

/// \brief Dispatches vector multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class VecX, class VecV>
VecX& divide_assign(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	detail::divide_assign(x,v,typename VecV::evaluation_category());
	return x();
}
	
/////////////////////////////////////////////////////////////////////////////////////
////// Matrix Assign
////////////////////////////////////////////////////////////////////////////////////
	
namespace detail{
	template<class MatA, class MatB>
	void assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,elementwise_tag){
		kernels::assign(A,B);
	}
	template<class MatA, class MatB>
	void assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,blockwise_tag){
		B().assign_to(A);
	}
	template<class MatA, class MatB>
	void plus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,elementwise_tag){
		kernels::assign<scalar_plus_assign> (A, B);
	}
	template<class MatA, class MatB>
	void plus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,blockwise_tag){
		B().plus_assign_to(A);
	}
	template<class MatA, class MatB>
	void minus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,elementwise_tag){
		kernels::assign<scalar_minus_assign> (A, B);
	}
	template<class MatA, class MatB>
	void minus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,blockwise_tag){
		B().minus_assign_to(A);
	}
	template<class MatA, class MatB>
	void multiply_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,elementwise_tag){
		kernels::assign<scalar_multiply_assign> (A, B);
	}
	template<class MatA, class MatB>
	void multiply_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,blockwise_tag){
		typename matrix_temporary<MatA>::type temporary(B);
		kernels::assign<scalar_multiply_assign> (A, B);
	}
	template<class MatA, class MatB>
	void divide_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,elementwise_tag){
		kernels::assign<scalar_divide_assign> (A, B);
	}
	template<class MatA, class MatB>
	void divide_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B,blockwise_tag){
		typename matrix_temporary<MatA>::type temporary(B);
		kernels::assign<scalar_divide_assign> (A, B);
	}
}
	

/// \brief Dispatches matrix assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB>
MatA& assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	detail::assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB>
MatA& plus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	detail::plus_assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix plus-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB>
MatA& minus_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	detail::minus_assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix multiply-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB>
MatA& multiply_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	detail::multiply_assign(A,B, typename MatB::evaluation_category());
	return A();
}

/// \brief Dispatches matrix divide-assignment on an expression level
///
/// This dispatcher takes care for whether the blockwise evaluation
/// or the elementwise evaluation is called
template<class MatA, class MatB>
MatA& divide_assign(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	detail::divide_assign(A,B, typename MatB::evaluation_category());
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
template<class VecX, class VecV>
VecX& operator+=(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return plus_assign(x,temporary);
}

/// \brief  Subtract-Assigns two vector expressions
///
/// Performs the operation x_i-=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)-=v to avoid this if A and B do not alias
template<class VecX, class VecV>
VecX& operator-=(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return minus_assign(x,temporary);
}

/// \brief  Multiply-Assigns two vector expressions
///
/// Performs the operation x_i*=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)*=v to avoid this if A and B do not alias
template<class VecX, class VecV>
VecX& operator*=(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return multiply_assign(x,temporary);
}

/// \brief  Divide-Assigns two vector expressions
///
/// Performs the operation x_i/=v_i for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(x)/=v to avoid this if A and B do not alias
template<class VecX, class VecV>
VecX& operator/=(vector_expression<VecX>& x, vector_expression<VecV> const& v){
	SIZE_CHECK(x().size() == v().size());
	typename vector_temporary<VecX>::type temporary(v);
	return divide_assign(x,temporary);
}

/// \brief  Adds a scalar to all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX>
VecX& operator+=(vector_expression<VecX>& x, typename VecX::scalar_type t){
	kernels::assign<scalar_plus_assign> (x, t);
	return x();
}

/// \brief  Subtracts a scalar from all elements of the vector
///
/// Performs the operation x_i += t for all elements.
template<class VecX>
VecX& operator-=(vector_expression<VecX>& x, typename VecX::scalar_type t){
	kernels::assign<scalar_minus_assign> (x, t);
	return x();
}

/// \brief  Multiplies a scalar with all elements of the vector
///
/// Performs the operation x_i *= t for all elements.
template<class VecX>
VecX& operator*=(vector_expression<VecX>& x, typename VecX::scalar_type t){
	kernels::assign<scalar_multiply_assign> (x, t);
	return x();
}

/// \brief  Divides all elements of the vector by a scalar
///
/// Performs the operation x_i /= t for all elements.
template<class VecX>
VecX& operator/=(vector_expression<VecX>& x, typename VecX::scalar_type t){
	kernels::assign<scalar_divide_assign> (x, t);
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
template<class MatA, class MatB>
MatA& operator+=(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return plus_assign(A,temporary);
}

/// \brief  Subtract-Assigns two matrix expressions
///
/// Performs the operation A_ij-=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)-=B to avoid this if A and B do not alias
template<class MatA, class MatB>
MatA& operator-=(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return minus_assign(A,temporary);
}

/// \brief  Multiply-Assigns two matrix expressions
///
/// Performs the operation A_ij*=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)*=B to avoid this if A and B do not alias
template<class MatA, class MatB>
MatA& operator*=(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return multiply_assign(A,temporary);
}

/// \brief  Divide-Assigns two matrix expressions
///
/// Performs the operation A_ij/=B_ij for all elements.
/// Assumes that the right and left hand side aliases and therefore 
/// performs a copy of the right hand side before assigning
/// use noalias as in noalias(A)/=B to avoid this if A and B do not alias
template<class MatA, class MatB>
MatA& operator/=(matrix_expression<MatA>& A, matrix_expression<MatB> const& B){
	SIZE_CHECK(A().size1() == B().size1());
	SIZE_CHECK(A().size2() == B().size2());
	typename matrix_temporary<MatA>::type temporary(B);
	return divide_assign(A,temporary);
}

/// \brief  Adds a scalar to all elements of the matrix
///
/// Performs the operation A_ij += t for all elements.
template<class MatA>
MatA& operator+=(matrix_expression<MatA>& A, typename MatA::scalar_type t){
	kernels::assign<scalar_plus_assign> (A, t);
	return A();
}

/// \brief  Subtracts a scalar from all elements of the matrix
///
/// Performs the operation A_ij -= t for all elements.
template<class MatA>
MatA& operator-=(matrix_expression<MatA>& A, typename MatA::scalar_type t){
	kernels::assign<scalar_minus_assign> (A, t);
	return A();
}

/// \brief  Multiplies a scalar to all elements of the matrix
///
/// Performs the operation A_ij *= t for all elements.
template<class MatA>
MatA& operator*=(matrix_expression<MatA>& A, typename MatA::scalar_type t){
	kernels::assign<scalar_multiply_assign> (A, t);
	return A();
}

/// \brief  Divides all elements of the matrix by a scalar
///
/// Performs the operation A_ij /= t for all elements.
template<class MatA>
MatA& operator /=(matrix_expression<MatA>& A, typename MatA::scalar_type t){
	kernels::assign<scalar_divide_assign> (A, t);
	return A();
}



//////////////////////////////////////////////////////////////////////////////////////
///// Temporary Proxy Operators
/////////////////////////////////////////////////////////////////////////////////////

template<class T, class U>
temporary_proxy<T> operator+=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) += arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator-=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) -= arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator*=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) *= arg;
	return x;
}
template<class T, class U>
temporary_proxy<T> operator/=(temporary_proxy<T> x, U const& arg){
	static_cast<T&>(x) /= arg;
	return x;
}




// Assignment proxy.
// Provides temporary free assigment when LHS has no alias on RHS
template<class C>
class noalias_proxy{
public:
	typedef typename C::closure_type closure_type;
	typedef typename C::scalar_type scalar_type;

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
	closure_type &operator+= (scalar_type t) {
		return m_lval += t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)-=2;
	closure_type &operator-= (scalar_type t) {
		return m_lval -= t;
	}
	
	//this is not needed, but prevents errors when for example doing noalias(x)*=2;
	closure_type &operator*= (scalar_type t) {
		return m_lval *= t;
	}

	//this is not needed, but prevents errors when for example doing noalias(x)/=2;
	closure_type &operator/= (scalar_type t) {
		return m_lval /= t;
	}

private:
	closure_type m_lval;
};

// Improve syntax of efficient assignment where no aliases of LHS appear on the RHS
//  noalias(lhs) = rhs_expression
template <class C>
noalias_proxy<C> noalias(matrix_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(vector_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}

template <class C>
noalias_proxy<C> noalias(matrix_set_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(vector_set_expression<C>& lvalue) {
	return noalias_proxy<C> (lvalue());
}
template <class C>
noalias_proxy<C> noalias(temporary_proxy<C> lvalue) {
	return noalias_proxy<C> (static_cast<C&>(lvalue));
}




//////////////////////////////////////////////////////////////////////
/////Evaluate blockwise expressions
//////////////////////////////////////////////////////////////////////
namespace detail{
	template<class E>
	E const& evaluate_block(
		blas::vector_expression<E> const& e,
		elementwise_tag
	){
		return e();
	}
	template<class E>
	typename vector_temporary<E>::type evaluate_block(
		blas::vector_expression<E> const& e,
		blockwise_tag
	){
		return e();
	}
	template<class E>
	E const& evaluate_block(
		blas::matrix_expression<E> const& e,
		elementwise_tag
	){
		return e();
	}
	template<class E>
	typename matrix_temporary<E>::type evaluate_block(
		blas::matrix_expression<E> const& e,
		blockwise_tag
	){
		return e();
	}
}

///\brief conditionally evaluates a vector expression if it is a block expression
///
/// If the expression is a block expression, a temporary vector is created to which
/// the expression is assigned, which is then returned, otherwise the expression itself
/// is returned
template<class E>
typename boost::mpl::eval_if<
	boost::is_same<
		typename E::evaluation_category,
		blockwise_tag
	>,
	vector_temporary<E>,
	boost::mpl::identity<E const&>
>::type
eval_block(blas::vector_expression<E> const& e){
	return detail::evaluate_block(e,typename E::evaluation_category());
}
///\brief conditionally evaluates a matrix expression if it is a block expression
///
/// If the expression is a block expression, a temporary matrix is created to which
/// the expression is assigned, which is then returned, otherwise the expression itself
/// is returned
template<class E>
typename boost::mpl::eval_if<
	boost::is_same<
		typename E::evaluation_category,
		blockwise_tag
	>,
	matrix_temporary<E>,
	boost::mpl::identity<E const&>
>::type
eval_block(blas::matrix_expression<E> const& e){
	return detail::evaluate_block(e,typename E::evaluation_category());
}

}}
#endif