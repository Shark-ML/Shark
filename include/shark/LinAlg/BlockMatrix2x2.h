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


#ifndef SHARK_LINALG_BLOCKMATRIX2X2_H
#define SHARK_LINALG_BLOCKMATRIX2X2_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {


///
/// \brief SVM regression matrix
///
/// \par
/// The BlockMatrix2x2 class is a \f$ 2n \times 2n \f$ block matrix of the form<br>
/// &nbsp;&nbsp;&nbsp; \f$ \left( \begin{array}{lr} M & M \\ M & M \end{array} \right) \f$ <br>
/// where M is an \f$ n \times n \f$ matrix.
/// This matrix form is needed in SVM regression problems.
///
template <class Matrix>
class BlockMatrix2x2
{
public:
    typedef typename Matrix::QpFloatType QpFloatType;

    /// Constructor.
    /// \param base  underlying matrix M, see class description of BlockMatrix2x2.
    BlockMatrix2x2(Matrix* base)
    {
        m_base = base;

        m_mapping.resize(size());

        std::size_t ic = m_base->size();
        for (std::size_t i = 0; i < ic; i++)
        {
            m_mapping[i] = i;
            m_mapping[i + ic] = i;
        }
    }


    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        return m_base->entry(m_mapping[i], m_mapping[j]);
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t i, std::size_t start,std::size_t end, QpFloatType* storage) const{
        for(std::size_t j = start; j < end; j++){
            storage[j-start] = m_base->entry(m_mapping[i], m_mapping[j]);
        }
    }
    
    /// \brief Computes the kernel-matrix
    template<class M>
    void matrix(
        blas::matrix_expression<M> & storage
    ) const{
        for(std::size_t i = 0; i != size(); ++i){
            for(std::size_t j = 0; j != size(); ++j){
                storage()(i,j) = entry(i,j);
            }
        }
    }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j)
    {
        std::swap(m_mapping[i], m_mapping[j]);
    }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return 2 * m_base->size(); }

protected:
    /// underlying KernelMatrix object
    Matrix* m_base;

    /// coordinate permutation
    std::vector<std::size_t> m_mapping;
};

}
#endif
