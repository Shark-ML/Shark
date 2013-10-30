//===========================================================================
/*!
 *  \file QuadraticProgram.h
 *
 *  \brief Quadratic programming for Support Vector Machines
 *
 *
 *  \par
 *  This file provides a number of classes representing hugh dense
 *  matrices all related to kernel Gram matices of possibly large
 *  datasets. These classes share a common interface for
 *     (a) providing a matrix entry,
 *     (b) swapping two variable indices, and
 *     (c) returning the matrix size.
 *
 *  \par
 *  This interface is required by the template class CachedMatrix,
 *  which provides a cache mechanism for restricted matrix rows, as it
 *  is used by various quadratic program solvers within the library.
 *  The PrecomputedMatrix provides a sometimes faster but more memory
 *  intensive alternative to CachedMatrix.
 *
 *
 *  \author  T. Glasmachers
 *  \date    2007-2012
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
//===========================================================================


#ifndef SHARK_ALGORITHMS_QP_QUADRATICPROGRAM_H
#define SHARK_ALGORITHMS_QP_QUADRATICPROGRAM_H

#include <shark/Algorithms/QP/LRUCache.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/OpenMP.h>

#include <boost/range/algorithm_ext/iota.hpp>
#include <vector>
#include <cmath>


namespace shark {

/// reason for the quadratic programming solver
/// to stop the iterative optimization process
enum QpStopType
{
	QpNone = 0,
	QpAccuracyReached = 1,
	QpMaxIterationsReached = 4,
	QpTimeout = 8,
};


///
/// \brief stopping conditions for quadratic programming
///
/// \par
/// The QpStoppingCondition structure defines conditions
/// for stopping the optimization procedure.
///
//! \par
//! For practical considerations the solvers supports
//! several stopping conditions. Usually, the optimization
//! stops if the Karush-Kuhn-Tucker (KKT) condition for
//! optimality are satisfied up to a certain accuracy.
//! In the case the optimal function value is known a
//! priori it is possible to stop as soon as the objective
//! is above a given threshold. In both cases it is very
//! difficult to predict the runtime. Therefore the
//! solver can stop after a predefined number of
//! iterations or after a predefined time period. In
//! these cases the solution found will not be near
//! optimal. In SVM training, using sensitive settings,
//! this should happen only during model selection while
//! investigating hyperparameters with poor
//! generalization ability.
//!
struct QpStoppingCondition
{
	/// Constructor
	QpStoppingCondition(double accuracy = 0.001, unsigned long long iterations = 0xffffffff, double value = 1e100, double seconds = 1e100)
	{
		minAccuracy = accuracy;
		maxIterations = iterations;
		targetValue = value;
		maxSeconds = seconds;
	}

	/// minimum accuracy to be achieved, usually KKT violation
	double minAccuracy;

	/// maximum number of decomposition iterations (default to 0 - not used)
	unsigned long long maxIterations;

	/// target objective function value (defaults to 1e100 - not used)
	double targetValue;

	/// maximum process time (defaults to 1e100 - not used)
	double maxSeconds;
};


///
/// \brief properties of the solution of a quadratic program
///
/// \par
/// The QpSolutionProperties structure collects statistics
/// about the approximate solutions found in a solver run.
/// It reports the reason for the iterative solver to stop,
/// which was set according to the QpStoppingCondition
/// structure. Furthermore it reports the solution accuracy,
/// the number of iterations, time elapsed, and the value
/// of the objective function in the reported solution.
///
struct QpSolutionProperties
{
	QpSolutionProperties()
	{
		type = QpNone;
		accuracy = 1e100;
		iterations = 0;
		value = 1e100;
		seconds = 0.0;
	}

	QpStopType type;						///< reason for the solver to stop
	double accuracy;						///< typically the maximal KKT violation
	unsigned long long iterations;			///< number of decomposition iterations
	double value;							///< value of the objective function
	double seconds;							///< training time
};


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

///\brief Specialization for dense vectors which often can be computed much faster
template <class T, class CacheType>
class KernelMatrix<blas::vector<T>, CacheType>
{
public:

	//////////////////////////////////////////////////////////////////
	// The types below define the type used for caching kernel values. The default is float,
	// since this type offers sufficient accuracy in the vast majority of cases, at a memory
	// cost of only four bytes. However, the type definition makes it easy to use double instead
	// (e.g., in case high accuracy training is needed).
	typedef CacheType QpFloatType;
	typedef blas::vector<T> InputType;

	/// Constructor
	/// \param kernelfunction   kernel function defining the Gram matrix
	/// \param data             data to evaluate the kernel function
	KernelMatrix(
		AbstractKernelFunction<InputType> const& kernelfunction,
		Data<InputType> const& data)
	: kernel(kernelfunction)
	, m_data(data)
	, m_batchStart(data.numberOfBatches())
	, m_accessCounter( 0 )
	{
		m_data.makeIndependent();
		std::size_t elements = m_data.numberOfElements();
		x.resize(elements);
		typename Data<InputType>::element_range::iterator iter=m_data.elements().begin();
		for(std::size_t i = 0; i != elements; ++i,++iter){
			x[i]=iter.getInnerIterator();
		}
		
		for(std::size_t i = 0,start = 0; i != m_data.numberOfBatches(); ++i){
			m_batchStart[i] = start;
			start+= m_data.batch(i).size1();
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
	void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const
	{
		m_accessCounter +=end-start;
		
		typename AbstractKernelFunction<InputType>::ConstInputReference xi = *x[k];		
		typename blas::matrix<T> mx(1,xi.size());
		noalias(blas::row(mx,0))=xi;
		
		int numBatches = (int)m_data.numberOfBatches();
		SHARK_PARALLEL_FOR(int i = 0; i < numBatches; i++)
		{
			std::size_t pos = m_batchStart[i];
			std::size_t batchSize = m_data.batch(i).size1();
			if(!(pos+batchSize < start || pos > end)){
				RealMatrix rowpart(1,batchSize);
				kernel.eval(mx,m_data.batch(i),rowpart);
				std::size_t batchStart = (start <=pos) ? 0: start-pos;
				std::size_t batchEnd = (pos+batchSize > end) ? end-pos: batchSize;
				for(std::size_t j =  batchStart;  j !=  batchEnd;++j){
					storage[pos+j-start] = static_cast<QpFloatType>(rowpart(0,j));
				}
			}
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
		if( i == j ) return;
		swap(*x[i],*x[j]);
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

	typedef typename Batch<InputType>::iterator PointerType;
	/// Array of data pointers for kernel evaluations
	std::vector<PointerType> x;

	std::vector<std::size_t> m_batchStart;

	mutable unsigned long long m_accessCounter;
};

///\brief Efficient special case if the kernel is gaussian and the inputs are sparse vectors
template <class T, class CacheType>
class GaussianKernelMatrix
{
public:

	typedef CacheType QpFloatType;
	typedef T InputType;

	/// Constructor
	/// \param gamma   bandwidth parameter of Gaussian kernel
	/// \param data    data evaluated by the kernel function
	GaussianKernelMatrix(
		double gamma,
		Data<InputType> const& data
	)
	: m_squaredNorms(data.numberOfElements())
	, m_gamma(gamma)
	, m_accessCounter( 0 )
	{
		std::size_t elements = data.numberOfElements();
		x.resize(elements);
		typename Data<InputType>::const_element_range::iterator iter=data.elements().begin();
		for(std::size_t i = 0; i != elements; ++i,++iter){
			x[i]=*iter;
			m_squaredNorms(i) =inner_prod(x[i],x[i]);//precompute the norms
		}
	}

	/// return a single matrix entry
	QpFloatType operator () (std::size_t i, std::size_t j) const
	{ return entry(i, j); }

	/// return a single matrix entry
	QpFloatType entry(std::size_t i, std::size_t j) const
	{
		++m_accessCounter;
		double distance = m_squaredNorms(i)-2*inner_prod(x[i], x[j])+m_squaredNorms(j);
		return (QpFloatType)std::exp(- m_gamma * distance);
	}
	
	/// \brief Computes the i-th row of the kernel matrix.
	///
	///The entries start,...,end of the i-th row are computed and stored in storage.
	///There must be enough room for this operation preallocated.
	void row(std::size_t i, std::size_t start,std::size_t end, QpFloatType* storage) const
	{
		m_accessCounter +=end-start;
		SHARK_PARALLEL_FOR(int j = start; j < (int) end; j++)
		{
			double distance = m_squaredNorms(i)-2*inner_prod(x[i], x[j])+m_squaredNorms(j);
			storage[j-start] = std::exp(- m_gamma * distance);
		}
	}
	
	/// \brief Computes the kernel-matrix
	template<class M>
	void matrix(
		blas::matrix_expression<M> & storage
	) const{
		for(std::size_t i = 0; i != size(); ++i){
			row(i,0,size(),&storage()(i,0));
		}
	}

	/// swap two variables
	void flipColumnsAndRows(std::size_t i, std::size_t j){
		using std::swap;
		swap(x[i],x[j]);
		swap(m_squaredNorms[i],m_squaredNorms[j]);
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

	typedef blas::sparse_vector_adaptor<typename T::value_type const,std::size_t> PointerType;
	/// Array of data pointers for kernel evaluations
	std::vector<PointerType> x;
	
	RealVector m_squaredNorms;

	double m_gamma;

	/// counter for the kernel accesses
	mutable unsigned long long m_accessCounter;
};

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


///
/// \brief Efficient quadratic matrix cache
///
/// \par
/// The access operations of the CachedMatrix class
/// are specially tuned towards the iterative solution
/// of quadratic programs resulting in sparse solutions.
///
/// \par
/// The kernel cache is probably one of the most intricate
/// or mind-twisting parts of Shark. In order to fully understand
/// it, reading the source code is essential and this description
/// naturally not sufficient. However, the general ideas are as
/// follows:
///
/// \par
/// A CachedMatrix owns a pointer to a regular (non-cached)
/// kernel matrix, the exact type of which is a template
/// parameter. Through it, the actions of requesting an entry
/// and propagating column-/row-flips are carried out. Even
/// though a CachedMatrix offers some methods also offered
/// by the general KernelMatrix, note that it does not inherit
/// from it in order to offer greater flexibility.
///
/// \par
/// The CachedMatrix defines a struct tCacheEntry, which
/// represents one row of varying length of a stored kernel matrix.
/// The structure aggregates a pointer to the kernel values (stored
/// as values of type CacheType); the number of stored values; and
/// the indices of the next older and newer cache entries. The latter
/// two indices pertain to the fact that the CachedMatrix maintains
/// two different "orders" of the examples: one related to location
/// in memory, and the other related to last usage time, see below.
/// During the lifetime of a CachedMatrix, it will hold a vector of
/// the length of the number of examples of tCacheEntry: one entry
/// for each example. When an example has no kernel values in the cache,
/// its tCacheEntry.length will be 0, its tCacheEntry.data will be NULL,
/// and its older and newer variables will be SHARK_CACHEDMATRIX_NULL_REFERENCE.
/// Otherwise, the entries take their corresponding meaningful values.
/// In particular for tCacheEntry.data, memory is dynamically allocated
/// via malloc, reallocated via realloc, and freed via free as fit.
///
/// \par
/// A basic operation the CachedMatrix must support is throwing away
/// old stored values to make room for new values, because it is very
/// common that not all values fit into memory (otherwise, consider the
/// PrecomputedMatrix class). When a new row is requested via row(..),
/// but no room to store it, row(..) has two options for making space:
///
/// \par
/// First option: first, the row method checks if the member index
/// m_truncationColumnIndex is lower than the overall number of examples.
/// If so, it goes through all existing rows and shortens them to a length
/// of m_truncationColumnIndex. Through this shortening, memory becomes
/// available. In other words, m_truncationColumnIndex can be used to
/// indicate to the CachedMatrix that every row longer than
/// m_truncationColumnIndex can be clipped at the end. By default,
/// m_truncationColumnIndex is equal to the number of examples and not
/// changed by the CachedMatrix, so no clipping will occur if the
/// CachedMatrix is left to its own devices. However, m_truncationColumnIndex
/// can be set from externally via setTruncationIndex(..) [this might be
/// done after a shrinking event, for example]. Now imagine a situation
/// where the cache is full, and the possibility exists to free some
/// memory by truncating longer cache rows to length m_truncationColumnIndex.
/// As soon as enough rows have been clipped for a new row to fit in, the
/// CachedMatrix computes the new row and passes back control. Most likely,
/// the next time a new, uncached row is requested, more rows will have to
/// get clipped. In order not to start checking if rows can be clipped from
/// the very first row over again, the member variable m_truncationRowIndex
/// internally stores where the chopping-procedure left off the last time.
/// When a new row is requested and it's time to clear out old entries, it
/// will start looking for choppable rows at this index to save time. In
/// general, any chopping would happen via cacheResize(..) internally.
///
/// \par
/// Second option: if all rows have been chopped of at the end, or if this
/// has never been an option (due to all rows being shorter or equal to
/// m_truncationColumnIndex anyways), entire rows will get discarded as
/// the second option. This will probably be the more common case. In
/// general, row deletions will happen via cacheDelete(..) internally.
/// The CachedMatrix itself only resorts to a very simple heuristic in
/// order to determine which rows to throw away to make room for new ones.
/// Namely, the CachedMatrix keeps track of the "age" or "oldness" of all
/// cached rows. This happens via the so-to-speak factually doubly-linked
/// list of indices in the tCacheEntry.older/newer entries, plus two class
/// members m_cacheNewest and m_cacheOldest, which point to the beginning
/// and end of this list. When row(..) wants to delete a cached row, it
/// always does so on the row with index m_cacheOldest, and this index is
/// then set to the next oldest row. Likewise, whenever a new row is requested,
/// m_cacheNewest is set to point to that one. In order to allow for smarter
/// heuristics, external classes may intervene with the deletion order via
/// the methods cacheRedeclareOldest and cacheRedeclareNewest, which move
/// an example to be deleted next or as late as possible, respectively.
///
/// \par
/// Two more drastic possibilites to influence the cache behaviour are
/// cacheRowResize and cacheRowRelease, which both directly resize (e.g.,
/// chop off) cached row values or even discard the row altogether.
/// In general however, it is preferred that the external application
/// only indicate preferences for deletion, because it will usually not
/// have information on the fullness of the cache (although this functionality
/// could easily be added).
///
template <class Matrix>
class CachedMatrix
{
public:
	typedef typename Matrix::QpFloatType QpFloatType;

	/// Constructor
	/// \param base       Matrix to cache
	/// \param cachesize  Main memory to use as a kernel cache, in QpFloatTypes. Default is 256MB if QpFloatType is float, 512 if double.
	CachedMatrix(Matrix* base, std::size_t cachesize = 0x4000000)
	: mep_baseMatrix(base), m_cache( base->size(),cachesize ){}
		
	/// \brief Copies the range [start,end) of the k-th row of the matrix in external storage
	///
	/// This call regards the access to the line as out-of-order, thus the cache is not influenced.
	/// \param k the index of the row
	/// \param start the index of the first element in the range
	/// \param end the index of the last element in the range
	/// \param storage the external storage. must be big enough capable to hold the range
	void row(std::size_t k, std::size_t start,std::size_t end, QpFloatType* storage) const{
		SIZE_CHECK(start <= end);
		SIZE_CHECK(end <= size());
		std::size_t cached= m_cache.lineLength(k);
		if ( start < cached)//copy already available data into the temporary storage
		{
			QpFloatType const* line = m_cache.getLinePointer(k);
			std::copy(line + start, line+cached, storage);
		}
		//evaluate the remaining entries
		mep_baseMatrix->row(k,cached,end,storage+(cached-start));
	}

	/// \brief Return a subset of a matrix row
	///
	/// \par
	/// This method returns an array of QpFloatType with at least
	/// the entries in the interval [begin, end[ filled in.
	///
	/// \param k      matrix row
	/// \param start  first column to be filled in
	/// \param end    last column to be filled in +1
	QpFloatType* row(std::size_t k, std::size_t start, std::size_t end){
		(void)start;//unused
		//Save amount of entries already cached
		std::size_t cached= m_cache.lineLength(k);
		//create or extend cache line
		QpFloatType* line = m_cache.getCacheLine(k,end);
		if (end > cached)//compute entries not already cached
			mep_baseMatrix->row(k,cached,end,line+cached);
		return line;
	}

	/// return a single matrix entry
	QpFloatType operator () (std::size_t i, std::size_t j) const{ 
		return entry(i, j);
	}

	/// return a single matrix entry
	QpFloatType entry(std::size_t i, std::size_t j) const{
		return mep_baseMatrix->entry(i, j);
	}

	///
	/// \brief Swap the rows i and j and the columns i and j
	///
	/// \par
	/// It may be advantageous for caching to reorganize
	/// the column order. In order to keep symmetric matrices
	/// symmetric the rows are swapped, too. This corresponds
	/// to swapping the corresponding variables.
	///
	/// \param  i  first row/column index
	/// \param  j  second row/column index
	///
	void flipColumnsAndRows(std::size_t i, std::size_t j)
	{
		if(i == j)
			return;
		if (i > j)
			std::swap(i,j);

		// exchange all cache row entries
		for (std::size_t  k = 0; k < size(); k++)
		{
			std::size_t length = m_cache.lineLength(k);
			if(length <= i) continue;
			QpFloatType* line = m_cache.getLinePointer(k);//do not affect caching
			if (j < length)
				std::swap(line[i], line[j]);
			else // only one element is available from the cache
				line[i] = mep_baseMatrix->entry(k, j);
		}
		m_cache.swapLineIndices(i,j);
		mep_baseMatrix->flipColumnsAndRows(i, j);
	}

	/// return the size of the quadratic matrix
	std::size_t size() const
	{ return mep_baseMatrix->size(); }

	/// return the size of the kernel cache (in "number of QpFloatType-s")
	std::size_t getMaxCacheSize() const
	{ return m_cache.maxSize(); }

	/// get currently used size of kernel cache (in "number of QpFloatType-s")
	std::size_t getCacheSize() const
	{ return m_cache.size(); }

	/// get length of one specific currently cached row
	std::size_t getCacheRowSize(std::size_t k) const
	{ return m_cache.lineLength(k); }
	
	bool isCached(std::size_t k) const
	{ return m_cache.isCached(k); }
	
	///\brief Restrict the cached part of the matrix to the upper left nxn sub-matrix
	void setMaxCachedIndex(std::size_t n){
		SIZE_CHECK(n <=size());
		
		//truncate lines which are too long
		//~ m_cache.restrictLineSize(n);//todo: we can do that better, only resize if the memory is actually needed
		//~ for(std::size_t i = 0; i != n; ++i){
			//~ if(m_cache.lineLength(i) > n)
				//~ m_cache.resizeLine(i,n);
		//~ }
		for(std::size_t i = n; i != size(); ++i){//mark the lines for deletion which are not needed anymore
			m_cache.markLineForDeletion(i);
		}
	}

	/// completely clear/purge the kernel cache
	void clear()
	{ m_cache.clear(); }

protected:
	Matrix* mep_baseMatrix; ///< matrix to be cached

	LRUCache<QpFloatType> m_cache; ///< cache of the matrix lines
};


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
