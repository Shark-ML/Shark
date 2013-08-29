/**
*
*  \brief Optimized operations for Linear Algebra
*
*  \author O.Krause
*  \date 2011
*
*  \par Copyright (c) 1998-2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*
*/
#ifndef SHARK_LINALG_BLAS_FAST_OPERATIONS_H
#define SHARK_LINALG_BLAS_FAST_OPERATIONS_H

#include <shark/LinAlg/BLAS/ublas.h>

/**
* \ingroup shark_globals
* 
* @{
*/
namespace shark { namespace blas{
	///\brief Fast matrix/matrix product.
	///
	///Computes C= alpha* A*B +beta*C.
	template<class MatA,class MatB,class MatC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		matrix_expression<MatB> const & matB,
		matrix_expression<MatC>& matC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/matrix product.
	///
	///Computes C= alpha* A*B +beta*C.
	template<class MatA,class MatB,class MatC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		matrix_expression<MatB> const & matB,
		matrix_range<MatC> matC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/vector product.
	///
	///Computes c= alpha* A*b +beta*c.
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		vector_expression<VecB> const & vecB,
		vector_expression<VecC>& vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Special case of matrix/vector product for subrange result.
	///
	///Computes c= alpha* A*b +beta*c, where c is a subrange of a bigger vector
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		vector_expression<VecB> const & vecB,
		vector_range<VecC> vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Special case of matrix/vector product for matrix row results.
	///
	///Computes c= alpha* A*b +beta*c, where c is a matrix row
	template<class MatA,class VecB,class MatC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		vector_expression<VecB> const & vecB,
		matrix_row<MatC> vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Special case of matrix/vector product for matrix column results.
	///
	///Computes c= alpha* A*b +beta*c, where c is a column of a matrix
	template<class MatA,class VecB,class MatC>
	void fast_prod(
		matrix_expression<MatA> const & matA,
		vector_expression<VecB> const & vecB,
		matrix_column<MatC> vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/vector product.
	///
	///Computes c= alpha* b^TA +beta*c.
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		vector_expression<VecB> const & vecB,
		matrix_expression<MatA> const & matA,
		vector_expression<VecC>& vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/vector product.
	///
	///Computes c= alpha* b^TA +beta*c.
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		vector_expression<VecB> const & vecB,
		matrix_expression<MatA> const & matA,
		vector_range<VecC>& vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/vector product.
	///
	///Computes c= alpha* b^TA +beta*c.
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		vector_expression<VecB> const & vecB,
		matrix_expression<MatA> const & matA,
		matrix_row<VecC>& vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast matrix/vector product.
	///
	///Computes c= alpha* b^TA +beta*c.
	template<class MatA,class VecB,class VecC>
	void fast_prod(
		vector_expression<VecB> const & vecB,
		matrix_expression<MatA> const & matA,
		matrix_column<VecC>& vecC,
	bool beta=false,double alpha=1.0);
	
	///\brief Fast rank k update to a symmetric matrix
	///
	///Computes C= alpha* A*A^T + beta*C.
	///C must be symmetric, else the results will be wrong
	template<class MatA,class MatC>
	void symmRankKUpdate(
		matrix_expression<MatA> const & matA,
		matrix_expression<MatC>& matC,
	bool beta=false,double alpha=1.0);
	
	///\brief calculates \f$ b+=A^1+A^2+A^n \f$ where \f$A^i\f$ are the columns of A
	template<class MatA,class VecB>
	void sumColumns(matrix_expression<MatA> const& A, vector_container<VecB>& b);
	
	///\brief calculates \f$ b+=A_1+A_2+A_n \f$ where \f$A_i\f$ are the rows of A
	///
	///This is often used to calculate the mean value of a set of data stored in a matrix batch
	template<class MatA,class VecB>
	void sumRows(matrix_expression<MatA> const& A, vector_container<VecB>& b);
	
	///\brief calculates \f$ b+=A^1+A^2+A^n \f$ where \f$A^i\f$ are the columns of A
	///
	///returns the result as return value instead of a second argument
	template<class MatA>
	vector<typename MatA::value_type> sumColumns(matrix_expression<MatA> const& A){
		vector<typename MatA::value_type> result;
		sumColumns(A,result);
		return result;
	}
	
	///\brief calculates \f$ b+=A_1+A_2+A_n \f$ where \f$A_i\f$ are the rows of A
	///
	///returns the result as return value instead of a second argument
	template<class MatA>
	vector<typename MatA::value_type> sumRows(matrix_expression<MatA> const& A){
		vector<typename MatA::value_type> result;
		sumRows(A,result);
		return result;
	}
	
	template<class MatA>
	typename MatA::value_type sumElements(matrix_expression<MatA> const& A){
		return sum(sumRows(A));
	}
	
}}
/** @}*/

#include "Impl/fast_prod_matrix.inl"
#include "Impl/fast_prod_vector.inl"
#include "Impl/symmRankKUpdate.inl"
#include "Impl/sumMatrix.inl"
#endif
