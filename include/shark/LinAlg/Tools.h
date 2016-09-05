/*!
 * 
 *
 * \brief       Helper functions for linear algebra component.
 * 
 * 
 *
 * \author      O.Krause M.Thuma
 * \date        20102011
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
#ifndef SHARK_LINALG_TOOLS_H
#define SHARK_LINALG_TOOLS_H

namespace shark {
namespace blas{
	
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

}}
#endif
