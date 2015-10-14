//===========================================================================
/*!
 * 
 *
 * \brief       Kernel matrix which supports kernel evaluations on data with missing features. 
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


#ifndef SHARK_LINALG_EXAMPLEMODIFIEDKERNELMATRIX_H
#define SHARK_LINALG_EXAMPLEMODIFIEDKERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {


/// Kernel matrix which supports kernel evaluations on data with missing features. At the same time, the entry of the
/// Gram matrix between examples i and j can be multiplied by two scaling factors corresponding to
/// the examples i and j, respectively. To this end, this class holds a vector of as many scaling coefficients
/// as there are examples in the dataset.
/// @note: most of code in this class is borrowed from KernelMatrix by copy/paste, which is obviously terribly ugly.
/// We could/should refactor classes in this file as soon as possible.
template <typename InputType, typename CacheType>
class ExampleModifiedKernelMatrix
{
public:
    typedef CacheType QpFloatType;

    /// Constructor
    /// \param kernelfunction   kernel function defining the Gram matrix
    /// \param data             data to evaluate the kernel function
    ExampleModifiedKernelMatrix(
        AbstractKernelFunction<InputType> const& kernelfunction,
        Data<InputType> const& data)
    : kernel(kernelfunction)
    , m_accessCounter( 0 )
    {
        std::size_t elements = data.numberOfElements();
        x.resize(elements);
        boost::iota(x,data.elements().begin());
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j)
    { std::swap(x[i], x[j]); }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return x.size(); }

    /// query the kernel access counter
    unsigned long long getAccessCount() const
    { return m_accessCounter; }

    /// reset the kernel access counter
    void resetAccessCount()
    { m_accessCounter = 0; }

    /// return a single matrix entry
    /// Override the Base::entry(...)
    /// formula: \f$ K\left(x_i, x_j\right)\frac{1}{s_i}\frac{1}{s_j} \f$
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        // typedef typename InputType::value_type InputValueType;
        INCREMENT_KERNEL_COUNTER( m_accessCounter );
        SIZE_CHECK(i < size());
        SIZE_CHECK(j < size());

        return (QpFloatType)evalSkipMissingFeatures(
            kernel,
            *x[i],
            *x[j]) * (1.0 / m_scalingCoefficients[i]) * (1.0 / m_scalingCoefficients[j]);
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t i, std::size_t start,std::size_t end, QpFloatType* storage) const{
        for(std::size_t j = start; j < end; j++){
            storage[j-start] = entry(i,j);
        }
    }
    
    /// \brief Computes the kernel-matrix
    template<class M>
    void matrix(
        blas::matrix_expression<M> & storage
    ) const{
        for(std::size_t i = 0; i != size(); ++i){
            for(std::size_t j = 0; j != size(); ++j){
                storage(i,j) = entry(i,j);
            }
        }
    }

    void setScalingCoefficients(const RealVector& scalingCoefficients)
    {
        SIZE_CHECK(scalingCoefficients.size() == size());
        m_scalingCoefficients = scalingCoefficients;
    }

protected:

    /// Kernel function defining the kernel Gram matrix
    AbstractKernelFunction<InputType> const& kernel;

    typedef typename Data<InputType>::const_element_range::const_iterator PointerType;
    /// Array of data pointers for kernel evaluations
    std::vector<PointerType> x;
    /// counter for the kernel accesses
    mutable unsigned long long m_accessCounter;

private:

    /// The scaling coefficients
    RealVector m_scalingCoefficients;
};

}
#endif
