/**
*
*  \brief Helper functions for linear algebra component.
*
*  \author O.Krause M.Thuma
*  \date 2010-2011
*
*  \par Copyright (c) 1998-2007:
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
#ifndef SHARK_LINALG_BLAS_TOOLS_H
#define SHARK_LINALG_BLAS_TOOLS_H

//#include <shark/LinAlg/BLAS/VectorTransformations.h>
#include <shark/LinAlg/BLAS/Impl/repmat.h>
#include <shark/LinAlg/BLAS/traits/metafunctions.h>
namespace shark {
namespace blas{
namespace detail{
	//sparse expressions get just all non zero elements zeroed
	template<class Matrix>
	void zero(matrix_expression<Matrix>& mat,boost::mpl::true_){
		typedef typename Matrix::value_type Value;
		typedef typename Matrix::iterator1 Iter1;
		typedef typename Matrix::iterator2 Iter2;

		Iter1 end1 = mat().end1();
		for (Iter1 i1 = mat().begin1(); i1 != end1; ++i1){
			Iter2 end2 = i1.end();
			for (Iter2 i2 = i1.begin(); i2 != end2; ++i2){
				*i2 = Value(0.0);
			}
		}
	}
	template<class Vector>
	void zero(vector_expression<Vector>& vec,boost::mpl::true_){
		typedef typename Vector::value_type Value;
		typedef typename Vector::iterator Iter;

		Iter end = vec().end();
		for (Iter i = vec().begin(); i != end; ++i){
			*i = Value(0.0);
		}
	}
	
	//dense container can use clear() which tends to be a bit faster
	template<class Matrix>
	void zero(matrix_container<Matrix>& mat,boost::mpl::false_){
		mat().clear();
	}
	template<class Vector>
	void zero(vector_container<Vector>& vec,boost::mpl::false_){
		vec().clear();
	}
	
	//everything else gets explicitely zeroed
	template<class Matrix>
	void zero(matrix_expression<Matrix>& mat,boost::mpl::false_){
		typedef typename Matrix::value_type Value;
		noalias(mat()) = zero_matrix<Value>(mat().size1(),mat().size2());
	}
	template<class Vector>
	void zero(vector_expression<Vector>& vec,boost::mpl::false_){
		typedef typename Vector::value_type Value;
		noalias(vec()) = zero_vector<Value>(vec().size());
	}
	
	template<class Matrix>
	void ensureSize(matrix_expression<Matrix>& mat,std::size_t rows, std::size_t columns){
		SIZE_CHECK(mat().size1() == rows);
		SIZE_CHECK(mat().size2() == columns);
	}
	template<class Matrix>
	void ensureSize(matrix_container<Matrix>& mat,std::size_t rows, std::size_t columns){
		mat().resize(rows,columns);
	}
	template<class Vector>
	void ensureSize(vector_expression<Vector>& vec,std::size_t size){
		SIZE_CHECK(vec().size() == size);
	}
	template<class Vector>
	void ensureSize(vector_container<Vector>& vec,std::size_t size){
		vec().resize(size);
	}
}

/**
* \ingroup shark_globals
* 
* @{
*/

//////////////////////////////////////DIAG//////////////////////////////////////////////////

///\brief returns the diagonal of a constant square matrix as vector
///
///given a matrix 
///   (1 2 3)
///A =(4 5 6)
///   (7 8 9)
///
///diag(A) = (1,5,9)
template<class Matrix>
matrix_vector_range<Matrix const> diag(matrix_expression<Matrix> const& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<Matrix const> diagonal(mat(),Range(0,mat().size1()),Range(0,mat().size1()));
	return diagonal;
}

///\brief returns the diagonal of a square matrix as vector
///
///given a matrix 
///   (1 2 3)
///A =(4 5 6)
///   (7 8 9)
///
///diag(A) = (1,5,9)
template<class Matrix>
matrix_vector_range<Matrix> diag(matrix_expression<Matrix>& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	matrix_vector_range<Matrix> diagonal(mat(),Range(0,mat().size1()),Range(0,mat().size1()));
	return diagonal;
}
	
//////////////////////////////////////ZERO////////////////////////////////////////
	
///\brief Zeros a matrix. If it is sparse, the structure is preserved
template<class Matrix>
void zero(matrix_expression<Matrix>& mat){
	detail::zero(mat(),typename traits::IsSparse<Matrix>::type());
}
///\brief Zeros a matrix. If it is sparse, the structure is preserved
template<class Vector>
void zero(vector_expression<Vector>& vec){
	detail::zero(vec(),typename traits::IsSparse<Vector>::type());
}

///\brief Zeros a subrange of a matrix. If it is sparse, the structure is preserved
template<class Matrix>
void zero(matrix_range<Matrix> mat){
	detail::zero(mat,typename traits::IsSparse<Matrix>::type());
}
///\brief Zeros a subrange of a vector.  If it is sparse, the structure is preserved
template<class Vector>
void zero(vector_range<Vector> vec){
	detail::zero(vec,typename traits::IsSparse<Vector>::type());
}
///\brief Zeros a row of a matrix. If it is sparse, the structure is preserved
template<class Vector>
void zero(matrix_row<Vector> vec){
	detail::zero(vec,typename traits::IsSparse<Vector>::type());
}
///\brief Zeros a column of a matrix. If it is sparse, the structure is preserved
template<class Vector>
void zero(matrix_column<Vector> vec){
	detail::zero(vec,typename traits::IsSparse<Vector>::type());
}
	
	
//////////////////////////////////////NUMBER NONZEROS////////////////////////////////////

template<class V>
std::size_t nonzeroElements(vector_expression<V> const& vec){
	//~ return vec().nnz(); //does not work because proxies don't support nnz()
	std::size_t nnz = 0;//count it by hand.
	for(typename V::const_iterator pos = vec().begin(); pos != vec().end(); ++pos,++nnz);
	return nnz;
	
}
//~ template<class M>
//~ std::size_t nonzeroElements(matrix_expression<M> const& mat){
	//~ return mat().nnz();
//~ }

	
//////////////////////////////////////IDENTITY////////////////////////////////////////

///\brief Initializes the square matrix A to be the identity matrix
template<class Matrix>
void identity(matrix_expression<Matrix>& mat){
	SIZE_CHECK(mat().size1() == mat().size2());
	std::size_t m = mat().size1();
	zero(mat);
	for(std::size_t i = 0; i != m; ++i){
		mat()(i,i) = 1;
	}
}
	
//////////////////////////////////////ENSURE SIZE////////////////////////////////////////
	
///\brief Ensures that the matrix has the right size.
///
///Tries to resize mat. If the matrix expression can't be resized a debug assertion is thrown.
template<class Matrix>
void ensureSize(matrix_expression<Matrix>& mat,std::size_t rows, std::size_t columns){
	detail::ensureSize(mat(),rows,columns);
}
///\brief Ensures that the vector has the right size.
///
///Tries to resize vec. If the vector expression can't be resized a debug assertion is thrown.
template<class Vector>
void ensureSize(vector_expression<Vector>& vec,std::size_t size){
	detail::ensureSize(vec(),size);
}
	
//////////////////////////////////////MATRIX BLOCKS////////////////////////////////////////

///\brief partitions the matrix in 4 blocks defined by one splitting point (i,j).
///
/// the blocks are defined using the intervals [0,...,i), [i,...,mat.size1()]
/// and [0,...,j), [j,...,mat.size2()]
template<class Matrix>
class Blocking{
public:
	
	Blocking(Matrix& matrix,std::size_t i, std::size_t j)
	: m_upperLeft(subrange(matrix,0,i,0,j))
	, m_upperRight(subrange(matrix,0,i,j,matrix.size2()))
	, m_lowerLeft(subrange(matrix,i,matrix.size1(),0,j))
	, m_lowerRight(subrange(matrix,i,matrix.size1(),j,matrix.size2()))
	{}
		
	/// \brief Returns the lower left block of the matrix.
	matrix_range<Matrix> const& upperLeft()const{
		return m_upperLeft;
	}
	/// \brief Returns the upper right block of the matrix.
	matrix_range<Matrix> const& upperRight()const{
		return m_upperRight;
	}
	/// \brief Returns the lower left block of the matrix.
	matrix_range<Matrix> const& lowerLeft()const{
		return m_lowerLeft;
	}

	/// \brief Returns the lower right block of the matrix.
	matrix_range<Matrix> const& lowerRight()const{
		return m_lowerRight;
	}
	
	/// \brief Returns the lower left block of the matrix.
	matrix_range<Matrix>& upperLeft(){
		return m_upperLeft;
	}
	/// \brief Returns the upper right block of the matrix.
	matrix_range<Matrix>& upperRight(){
		return m_upperRight;
	}
	/// \brief Returns the lower left block of the matrix.
	matrix_range<Matrix>& lowerLeft(){
		return m_lowerLeft;
	}

	/// \brief Returns the lower right block of the matrix.
	matrix_range<Matrix>& lowerRight(){
		return m_lowerRight;
	}
	
	
private:
	matrix_range<Matrix> m_upperLeft;
	matrix_range<Matrix> m_upperRight;
	matrix_range<Matrix> m_lowerLeft;
	matrix_range<Matrix> m_lowerRight;
};




/// \brief Returns the i-th row of an upper triangular matrix excluding the elements right of the diagonal
template<class Matrix>
vector_range<matrix_row<Matrix const> >triangularRow(
	matrix_expression<Matrix> const& mat,
	std::size_t i
){
	matrix_row<Matrix const> matRow = row(mat(),i);
	return subrange(matRow,0,i);
}
/// \brief Returns the i-th row of an upper triangular matrix excluding the elements right of the diagonal
template<class Matrix>
vector_range<matrix_row<Matrix> > triangularRow(
	matrix_expression<Matrix>& mat,
	std::size_t i
){
	matrix_row<Matrix> matRow = row(mat(),i);
	return subrange(matRow,0,i);
}
/// \brief Returns the elements in the i-th row of a lower triangular matrix left of the diagonal 
template<class Matrix>
vector_range<matrix_row<Matrix const> > unitTriangularRow(
	matrix_expression<Matrix> const& mat,
	std::size_t i
){
	matrix_row<Matrix const> matRow = row(mat(),i);
	return subrange(matRow,0,i+1);
}
/// \brief Returns the elements in the i-th row of a lower triangular matrix left of the diagonal 
template<class Matrix>
vector_range<matrix_row<Matrix> > unitTriangularRow(
	matrix_expression<Matrix>& mat,
	std::size_t i
){
	matrix_row<Matrix> matRow = row(mat(),i);
	return subrange(matRow,0,i+1);
}



/// \brief Returns the elements in the i-th column of the matrix below the diagonal 
template<class Matrix>
vector_range<matrix_column<Matrix const> > unitTriangularColumn(
	matrix_expression<Matrix> const& mat,
	std::size_t i
){
	matrix_column<Matrix const> col = column(mat(),i);
	return subrange(col,i+1,mat().size2());
}
/// \brief Returns the elements in the i-th column of the matrix below the diagonal 
template<class Matrix>
vector_range<matrix_column<Matrix> > unitTriangularColumn(
	matrix_expression<Matrix>& mat,
	std::size_t i
){
	matrix_column<Matrix> col = column(mat(),i);
	return subrange(col,i+1,mat().size2());
}

/// \brief Returns the elements in the i-th column of the matrix excluding the zero elements
template<class Matrix>
vector_range<matrix_column<Matrix const> > triangularColumn(
	matrix_expression<Matrix> const& mat,
	std::size_t i
){
	matrix_column<Matrix const> col = column(mat(),i);
	return subrange(col,i,mat().size2());
}
/// \brief Returns the elements in the i-th column of the matrix excluding the zero elements
template<class Matrix>
vector_range<matrix_column<Matrix> > triangularColumn(
	matrix_expression<Matrix>& mat,
	std::size_t i
){
	matrix_column<Matrix> col = column(mat(),i);
	return subrange(col,i,mat().size2());
}


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
VectorRepeater<Vector> repeat(vector_expression<Vector> const& vector, std::size_t rows){
	return VectorRepeater<Vector>(vector(),rows);
}

///brief repeats a single element to form a vector of length elements
///
///@param scalar the value which is repeated
///@param elements the size of the resulting vector
template<class T>
typename boost::enable_if<boost::is_arithmetic<T>, scalar_vector<T> >::type
repeat(T scalar, std::size_t elements){
	return scalar_vector<T>(elements,scalar);
}

///brief repeats a single element to form a matrix  of size rows x columns
///
///@param scalar the value which is repeated
///@param rows the number of rows of the resulting vector
///@param columns the number of columns of the resulting vector
template<class T>
typename boost::enable_if<boost::is_arithmetic<T>, scalar_matrix<T> >::type
repeat(T scalar, std::size_t rows, std::size_t columns){
	return scalar_matrix<T>(rows, columns, scalar);
}

///brief picks a subrange of rows from a matrix. much easier to use than subrange
template<class Matrix>
matrix_range<Matrix const> rows(matrix_expression<Matrix> const& mat,std::size_t start, std::size_t end){
	return subrange(mat(),start,end,0,mat().size2());
}
///brief picks a subrange of rows from a matrix. much easier to use than subrange
template<class Matrix>
matrix_range<Matrix> rows(matrix_expression<Matrix>& mat,std::size_t start, std::size_t end){
	return subrange(mat(),start,end,0,mat().size2());
}

///brief picks a subrange of columns from a matrix. much easier to use than subrange
template<class Matrix>
matrix_range<Matrix const> columns(matrix_expression<Matrix> const& mat,std::size_t start, std::size_t end){
	return subrange(mat(),0,mat().size1(),start,end);
}

///brief picks a subrange of columns from a matrix. much easier to use than subrange
template<class Matrix>
matrix_range<Matrix> columns(matrix_expression<Matrix>& mat,std::size_t start, std::size_t end){
	return subrange(mat(),0,mat().size1(),start,end);
}

//////////////////////////////////MISC//////////////////////////////////////////////////
	
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
 *      \param  v square matrix
 *      \return the sum of the values at the diagonal of \em v
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
/** @}*/
}}
#endif
