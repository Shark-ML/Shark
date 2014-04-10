//===========================================================================
/*!
 * 
 *
 * \brief       Kernel Gram matrix
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


#ifndef SHARK_LINALG_KERNELMATRIX_H
#define SHARK_LINALG_KERNELMATRIX_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>

#include <vector>
#include <cmath>


namespace shark {

///
/// \brief Kernel Gram matrix
///
/// \par
/// The KernelMatrix is the most prominent type of matrix
/// for quadratic programming. It provides the Gram matrix
/// of a fixed data set with respect to an inner product
/// implicitly defined by a kernel function.
///
/// \par
/// NOTE: The KernelMatrix class stores pointers to the
/// data, instead of maintaining a copy of the data. Thus,
/// it implicitly assumes that the dataset is not altered
/// during the lifetime of the KernelMatrix object. This
/// condition is ensured as long as the class is used via
/// the various SVM-trainers.
///
template <class InputType, class CacheType>
class KernelMatrix
{
public:
    typedef CacheType QpFloatType;

    /// Constructor
    /// \param kernelfunction   kernel function defining the Gram matrix
    /// \param data             data to evaluate the kernel function
    KernelMatrix(AbstractKernelFunction<InputType> const& kernelfunction,
            Data<InputType> const& data)
    : kernel(kernelfunction)
    , m_data(data)
    , m_accessCounter( 0 )
    {
        std::size_t elements = m_data.numberOfElements();
        x.resize(elements);
        typename Data<InputType>::const_element_range::iterator iter=m_data.elements().begin();
        for(std::size_t i = 0; i != elements; ++i,++iter){
            x[i]=iter.getInnerIterator();
        }
    }

    /// return a single matrix entry
    QpFloatType operator () (std::size_t i, std::size_t j) const
    { return entry(i, j); }

    /// return a single matrix entry
    QpFloatType entry(std::size_t i, std::size_t j) const
    {
        ++m_accessCounter;
        return (QpFloatType)kernel.eval(*x[i], *x[j]);
    }
    
    /// \brief Computes the i-th row of the kernel matrix.
    ///
    ///The entries start,...,end of the i-th row are computed and stored in storage.
    ///There must be enough room for this operation preallocated.
    void row(std::size_t i, std::size_t start,std::size_t end, QpFloatType* storage) const{
        m_accessCounter += end-start;
        
        typename AbstractKernelFunction<InputType>::ConstInputReference xi = *x[i];
        SHARK_PARALLEL_FOR(int j = start; j < (int) end; j++)
        {
            storage[j-start] = QpFloatType(kernel.eval(xi, *x[j]));
        }
    }
    
    /// \brief Computes the kernel-matrix
    template<class M>
    void matrix(
        blas::matrix_expression<M> & storage
    ) const{
        calculateRegularizedKernelMatrix(kernel,m_data,storage);
    }

    /// swap two variables
    void flipColumnsAndRows(std::size_t i, std::size_t j){
        using std::swap;
        swap(x[i],x[j]);
    }

    /// return the size of the quadratic matrix
    std::size_t size() const
    { return x.size(); }

    /// query the kernel access counter
    unsigned long long getAccessCount() const
    { return m_accessCounter; }

    /// reset the kernel access counter
    void resetAccessCount()
    { m_accessCounter = 0; }

protected:
    /// Kernel function defining the kernel Gram matrix
    const AbstractKernelFunction<InputType>& kernel;

    Data<InputType> m_data;

    typedef typename Batch<InputType>::const_iterator PointerType;
    /// Array of data pointers for kernel evaluations
    std::vector<PointerType> x;

    /// counter for the kernel accesses
    mutable unsigned long long m_accessCounter;
};

//~ ///\brief Specialization for dense vectors which often can be computed much faster
//~ template <class T, class CacheType>
//~ class KernelMatrix<blas::vector<T>, CacheType>
//~ {
//~ public:

    //~ //////////////////////////////////////////////////////////////////
    //~ // The types below define the type used for caching kernel values. The default is float,
    //~ // since this type offers sufficient accuracy in the vast majority of cases, at a memory
    //~ // cost of only four bytes. However, the type definition makes it easy to use double instead
    //~ // (e.g., in case high accuracy training is needed).
    //~ typedef CacheType QpFloatType;
    //~ typedef blas::vector<T> InputType;

    //~ /// Constructor
    //~ /// \param kernelfunction   kernel function defining the Gram matrix
    //~ /// \param data             data to evaluate the kernel function
    //~ KernelMatrix(
        //~ AbstractKernelFunction<InputType> const& kernelfunction,
        //~ Data<InputType> const& data)
    //~ : kernel(kernelfunction)
    //~ , m_data(data)
    //~ , m_batchStart(data.numberOfBatches())
    //~ , m_accessCounter( 0 )
    //~ {
        //~ m_data.makeIndependent();
        //~ std::size_t elements = m_data.numberOfElements();
        //~ x.resize(elements);
        //~ typename Data<InputType>::element_range::iterator iter=m_data.elements().begin();
        //~ for(std::size_t i = 0; i != elements; ++i,++iter){
            //~ x[i]=iter.getInnerIterator();
        //~ }
        
        //~ for(std::size_t i = 0,start = 0; i != m_data.numberOfBatches(); ++i){
            //~ m_batchStart[i] = start;
            //~ start+= m_data.batch(i).size1();
        //~ }
    //~ }

    //~ /// return a single matrix entry
    //~ QpFloatType operator () (std::size_t i, std::size_t j) const
    //~ { return entry(i, j); }

    //~ /// return a single matrix entry
    //~ QpFloatType entry(std::size_t i, std::size_t j) const
    //~ {
        //~ ++m_accessCounter;
        //~ return (QpFloatType)kernel.eval(*x[i], *x[j]);
    //~ }
    
    //~ /// \brief Computes the i-th row of the kernel matrix.
    //~ ///
    //~ ///The entries start,...,end of the i-th row are computed and stored in storage.
    //~ ///There must be enough room for this operation preallocated.
    //~ void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const
    //~ {
        //~ m_accessCounter +=end-start;
        
        //~ typename AbstractKernelFunction<InputType>::ConstInputReference xi = *x[k];     
        //~ typename blas::matrix<T> mx(1,xi.size());
        //~ noalias(blas::row(mx,0))=xi;
        
        //~ int numBatches = (int)m_data.numberOfBatches();
        //~ SHARK_PARALLEL_FOR(int i = 0; i < numBatches; i++)
        //~ {
            //~ std::size_t pos = m_batchStart[i];
            //~ std::size_t batchSize = m_data.batch(i).size1();
            //~ if(!(pos+batchSize < start || pos > end)){
                //~ RealMatrix rowpart(1,batchSize);
                //~ kernel.eval(mx,m_data.batch(i),rowpart);
                //~ std::size_t batchStart = (start <=pos) ? 0: start-pos;
                //~ std::size_t batchEnd = (pos+batchSize > end) ? end-pos: batchSize;
                //~ for(std::size_t j =  batchStart;  j !=  batchEnd;++j){
                    //~ storage[pos+j-start] = static_cast<QpFloatType>(rowpart(0,j));
                //~ }
            //~ }
        //~ }
    //~ }
    
    //~ /// \brief Computes the kernel-matrix
    //~ template<class M>
    //~ void matrix(
        //~ blas::matrix_expression<M> & storage
    //~ ) const{
        //~ calculateRegularizedKernelMatrix(kernel,m_data,storage);
    //~ }

    //~ /// swap two variables
    //~ void flipColumnsAndRows(std::size_t i, std::size_t j){
        //~ if( i == j ) return;
        //~ swap(*x[i],*x[j]);
    //~ }

    //~ /// return the size of the quadratic matrix
    //~ std::size_t size() const
    //~ { return x.size(); }
    
    //~ /// query the kernel access counter
    //~ unsigned long long getAccessCount() const
    //~ { return m_accessCounter; }

    //~ /// reset the kernel access counter
    //~ void resetAccessCount()
    //~ { m_accessCounter = 0; }

//~ protected:
    //~ /// Kernel function defining the kernel Gram matrix
    //~ const AbstractKernelFunction<InputType>& kernel;

    //~ Data<InputType> m_data;

    //~ typedef typename Batch<InputType>::iterator PointerType;
    //~ /// Array of data pointers for kernel evaluations
    //~ std::vector<PointerType> x;

    //~ std::vector<std::size_t> m_batchStart;

    //~ mutable unsigned long long m_accessCounter;
//~ };

}
#endif
