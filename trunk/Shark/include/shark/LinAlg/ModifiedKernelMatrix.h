//===========================================================================
/*!
 * 
 *
 * \brief       Modified Kernel Gram matrix
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
//===========================================================================


#ifndef SHARK_LINALG_MODIFIEDKERNELMATRIX_H
#define SHARK_LINALG_MODIFIEDKERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {

///
/// \brief Modified Kernel Gram matrix
///
/// \par
/// The ModifiedKernelMatrix represents the kernel matrix
/// multiplied element-wise with a factor depending on the
/// labels of the training examples. This is useful for the
/// MCMMR method (multi-class maximum margin regression).
template <class InputType, class CacheType>
class ModifiedKernelMatrix
{
private:
    typedef KernelMatrix<InputType,CacheType> Matrix;
public:
    typedef typename Matrix::QpFloatType QpFloatType;

    /// Constructor
    /// \param kernelfunction          kernel function
    /// \param data             data to evaluate the kernel function
    /// \param modifierEq multiplier for same-class labels
    /// \param modifierNe multiplier for different-class kernels
    ModifiedKernelMatrix(
        AbstractKernelFunction<InputType> const& kernelfunction,
        LabeledData<InputType, unsigned int> const& data,
        QpFloatType modifierEq,
        QpFloatType modifierNe
    ): m_matrix(kernelfunction,data.inputs())
    ,  m_labels(data.numberOfElements())
    , m_modifierEq(modifierEq)
    , m_modifierNe(modifierNe){
        for(std::size_t i = 0; i != m_labels.size(); ++i){
            m_labels[i] = data.element(i).label;
        }
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        QpFloatType ret = m_matrix(i,j);
        QpFloatType modifier = m_labels[i] == m_labels[j] ? m_modifierEq : m_modifierNe;
        return modifier*ret;
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t i, std::size_t start,std::size_t end, QpFloatType* storage) const{
        m_matrix.row(i,start,end,storage);
        //apply modifiers
        unsigned int labeli = m_labels[i];
        for(std::size_t j = start; j < end; j++){
            QpFloatType modifier = (labeli == m_labels[j]) ? m_modifierEq : m_modifierNe;
            storage[j-start] *= modifier;
        }
    }
    
    /// \brief Computes the kernel-matrix
    template<class M>
    void matrix(
        blas::matrix_expression<M> & storage
    ) const{
        m_matrix.matrix(storage);
        for(std::size_t i = 0; i != size(); ++i){
            unsigned int labeli = m_labels[i];
            for(std::size_t j = 0; j != size(); ++j){
                QpFloatType modifier = (labeli == m_labels[j]) ? m_modifierEq : m_modifierNe;
                storage()(i,j) *= modifier;
            }
        }
    }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j){
        m_matrix.flipColumnsAndRows(i,j);
        std::swap(m_labels[i],m_labels[j]);
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
    /// Kernel matrix which computes the basic entries.
    Matrix m_matrix;
    std::vector<unsigned int> m_labels;

    /// modifier in case the labels are equal
    QpFloatType m_modifierEq;

    /// modifier in case the labels differ
    QpFloatType m_modifierNe;
};


}
#endif
