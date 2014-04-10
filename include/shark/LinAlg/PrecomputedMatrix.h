//===========================================================================
/*!
 * 
 *
 * \brief       Precomputed version of a matrix for quadratic programming
 * 
 * 
 * \par
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
//===========================================================================


#ifndef SHARK_LINALG_PRECOMPUTEDMATRIX_H
#define SHARK_LINALG_PRECOMPUTEDMATRIX_H

#include <shark/Algorithms/QP/LRUCache.h>
#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {

///
/// \brief Precomputed version of a matrix for quadratic programming
///
/// \par
/// The PrecomputedMatrix class computes all pairs of kernel
/// evaluations in its constructor and stores them im memory.
/// This proceeding is only viable if the number of examples
/// does not exceed, say, about 10000. In this case the memory
/// demand is already \f$ 4 \cdot 10000^2 \approx 400\text{MB} \f$,
/// growing quadratically.
///
/// \par
/// Use of this class may be beneficial for certain model
/// selection strategies, in particular if the kernel is
/// fixed and the regularization parameter is varied.
///
/// \par
/// Use of this class may, in certain situations, even mean a
/// loss is speed, compared to CachedMatrix. This is the case
/// in particular if the solution of the quadratic program is
/// sparse, in which case most entries of the matrix do not
/// need to be computed at all, and the problem is "simple"
/// enough such that the solver's shrinking heuristic is not
/// mislead.
///
template <class Matrix>
class PrecomputedMatrix
{
public:
    typedef typename Matrix::QpFloatType QpFloatType;

    /// Constructor
    /// \param base  matrix to be precomputed
    PrecomputedMatrix(Matrix* base)
    : matrix(base->size(), base->size())
    {
        base->matrix(matrix);
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const{
        for(std::size_t j = start; j < end; j++){
            storage[j-start] = matrix(k, j);
        }
    }


    /// \brief Return a subset of a matrix row
    ///
    /// \par
    /// This method returns an array with at least
    /// the entries in the interval [begin, end[ filled in.
    ///
    /// \param k      matrix row
    /// \param begin  first column to be filled in
    /// \param end    last column to be filled in +1
    QpFloatType* row(std::size_t k, std::size_t begin, std::size_t end)
    {
        return &matrix(k, begin);
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        return matrix(i, j);
    }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j)
    {
        swap_rows(matrix,i, j);
        swap_columns(matrix,i, j);
    }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return matrix.size2(); }

    /// for compatibility with CachedMatrix
    std::size_t getMaxCacheSize()
    { return matrix.size1() * matrix.size2(); }

    /// for compatibility with CachedMatrix
    std::size_t getCacheSize() const
    { return getMaxCacheSize(); }

    /// for compatibility with CachedMatrix
    std::size_t getCacheRowSize(std::size_t k) const
    { return matrix.size2(); }
    
    /// for compatibility with CachedMatrix
    bool isCached(std::size_t){
        return true;
    }
    /// for compatibility with CachedMatrix
    void setMaxCachedIndex(std::size_t n){}
        
    /// for compatibility with CachedMatrix
    void clear()
    { }

protected:
    /// container for precomputed values
    blas::matrix<QpFloatType> matrix;
};

}
#endif
