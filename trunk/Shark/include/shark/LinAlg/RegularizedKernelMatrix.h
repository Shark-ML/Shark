//===========================================================================
/*!
 * 
 *
 * \brief       Kernel Gram matrix with modified diagonal
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


#ifndef SHARK_LINALG_REGULARIZEDKERNELMATRIX_H
#define SHARK_LINALG_REGULARIZEDKERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {


///
/// \brief Kernel Gram matrix with modified diagonal
///
/// \par
/// Regularized version of KernelMatrix. The regularization
/// is achieved by adding a vector to the matrix diagonal.
/// In particular, this is useful for support vector machines
/// with 2-norm penalty term.
///
template <class InputType, class CacheType>
class RegularizedKernelMatrix
{
private:
    typedef KernelMatrix<InputType,CacheType> Matrix;
public:
    typedef typename Matrix::QpFloatType QpFloatType;

    /// Constructor
    /// \param kernelfunction          kernel function
    /// \param data             data to evaluate the kernel function
    /// \param diagModification vector d of diagonal modifiers
    RegularizedKernelMatrix(
        AbstractKernelFunction<InputType> const& kernelfunction,
        Data<InputType> const& data,
        const RealVector& diagModification
    ):m_matrix(kernelfunction,data), m_diagMod(diagModification){
        SIZE_CHECK(size() == diagModification.size());
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        QpFloatType ret = m_matrix(i,j);
        if (i == j) ret += (QpFloatType)m_diagMod(i);
        return ret;
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const{
        m_matrix.row(k,start,end,storage);
        //apply regularization
        if(k >= start && k < end){
            storage[k-start] += (QpFloatType)m_diagMod(k);
        }
    }
    
    /// \brief Computes the kernel-matrix
    template<class M>
    void matrix(
        blas::matrix_expression<M> & storage
    ) const{
        m_matrix.matrix(storage);
        for(std::size_t k = 0; k != size(); ++k){
            storage()(k,k) += (QpFloatType)m_diagMod(k);
        }
    }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j){
        m_matrix.flipColumnsAndRows(i,j);
        std::swap(m_diagMod(i),m_diagMod(j));
    }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return m_matrix.size(); }

    /// query the kernel access counter
    unsigned long long getAccessCount() const
    { return m_matrix.getAccessCount(); }

    /// reset the kernel access counter
    void resetAccessCount()
    { m_matrix.resetAccessCount(); }

protected:
    Matrix m_matrix;
    RealVector m_diagMod;
};

}
#endif
