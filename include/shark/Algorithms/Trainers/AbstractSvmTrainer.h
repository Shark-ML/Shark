//===========================================================================
/*!
 * 
 *
 * \brief       Abstract Support Vector Machine Trainer, general and linear case
 * 
 * 
 * \par
 * This file provides: 1) the QpConfig class, which can configure and
 * provide information about an SVM training procedure; 2) a super-class
 * for general SVM trainers, namely the AbstractSvmTrainer; and 3) a
 * streamlined variant thereof for purely linear SVMs, namely the
 * AbstractLinearSvmTrainer. In general, the SvmTrainers hold as parameters
 * all hyperparameters of the underlying SVM, which includes the kernel
 * parameters for non-linear SVMs.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        -
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


#ifndef SHARK_ALGORITHMS_ABSTRACTSVMTRAINER_H
#define SHARK_ALGORITHMS_ABSTRACTSVMTRAINER_H


#include <shark/LinAlg/Base.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Algorithms/QP/QuadraticProgram.h>


namespace shark {


///
/// \brief Super class of all support vector machine trainers.
///
/// \par
/// The QpConfig class holds two structures describing
/// the stopping condition and the solution obtained by the underlying
/// quadratic programming solvers. It provides a uniform interface for
/// setting, e.g., the target solution accuracy and obtaining the
/// accuracy of the actual solution.
///
class QpConfig
{
public:
	/// Constructor
	QpConfig(bool precomputedFlag = false, bool sparsifyFlag = true)
	: m_precomputedKernelMatrix(precomputedFlag)
	, m_sparsify(sparsifyFlag)
	, m_shrinking(true)
	, m_s2do(true)
	, m_verbosity(0)
	, m_accessCount(0)
	{ }

	/// Read/write access to the stopping condition
	QpStoppingCondition& stoppingCondition()
	{ return m_stoppingcondition; }

	/// Read access to the stopping condition
	QpStoppingCondition const& stoppingCondition() const
	{ return m_stoppingcondition; }

	/// Access to the solution properties
	QpSolutionProperties& solutionProperties()
	{ return m_solutionproperties; }

	/// Flag for using a precomputed kernel matrix
	bool& precomputeKernel()
	{ return m_precomputedKernelMatrix; }

	/// Flag for using a precomputed kernel matrix
	bool const& precomputeKernel() const
	{ return m_precomputedKernelMatrix; }

	/// Flag for sparsifying the model after training
	bool& sparsify()
	{ return m_sparsify; }

	/// Flag for sparsifying the model after training
	bool const& sparsify() const
	{ return m_sparsify; }

	/// Flag for shrinking in the decomposition solver
	bool& shrinking()
	{ return m_shrinking; }

	/// Flag for shrinking in the decomposition solver
	bool const& shrinking() const
	{ return m_shrinking; }

	/// Flag for S2DO (instead of SMO)
	bool& s2do()
	{ return m_s2do; }

	/// Flag for S2DO (instead of SMO)
	bool const& s2do() const
	{ return m_s2do; }

	/// Verbosity level of the solver
	unsigned int& verbosity()
	{ return m_verbosity; }

	/// Verbosity level of the solver
	unsigned int const& verbosity() const
	{ return m_verbosity; }

	/// Number of kernel accesses
	unsigned long long const& accessCount() const
	{ return m_accessCount; }

	// Set threshold for minimum dual accuracy stopping condition
	void setMinAccuracy(double a) { m_stoppingcondition.minAccuracy = a; }
	// Set number of iterations for maximum number of iterations stopping condition
	void setMaxIterations(unsigned long long i) { m_stoppingcondition.maxIterations = i; }
	// Set values for target value stopping condition
	void setTargetValue(double v) { m_stoppingcondition.targetValue = v; }
	// Set maximum training time in seconds for the maximum seconds stopping condition
	void setMaxSeconds(double s) { m_stoppingcondition.maxSeconds = s; }
	
protected:
	/// conditions for when to stop the QP solver
	QpStoppingCondition m_stoppingcondition;
	/// properties of the approximate solution found by the solver
	QpSolutionProperties m_solutionproperties;
	/// should the solver use a precomputed kernel matrix?
	bool m_precomputedKernelMatrix;
	/// should the trainer sparsify the model after training?
	bool m_sparsify;
	/// should shrinking be used?
	bool m_shrinking;
	/// should S2DO be used instead of SMO?
	bool m_s2do;
	/// verbosity level (currently unused)
	unsigned int m_verbosity;
	/// kernel access count
	unsigned long long m_accessCount;
};


///
/// \brief Super class of all kernelized (non-linear) SVM trainers.
///
/// \par
/// This class holds general information shared by most if not
/// all SVM trainers. First of all, this includes the kernel and
/// the regularization parameter. The class also manages
/// meta-information of the training process, like the maximal
/// size of the kernel cache, the stopping criterion, as well
/// as information on the actual solution.
///
template <
	class InputType, class LabelType, 
	class Model = KernelClassifier<InputType>, 
	class Trainer= AbstractTrainer< Model,LabelType>
>
class AbstractSvmTrainer
: public Trainer,public QpConfig, public IParameterizable
{
public:
	typedef AbstractKernelFunction<InputType> KernelType;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param offset train svm with offset - this is not supported for all SVM solvers.
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	AbstractSvmTrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	: m_kernel(kernel)
	, m_regularizers(1,C)
	, m_trainOffset(offset)
	, m_unconstrained(unconstrained)
	, m_cacheSize(0x4000000)
	{ RANGE_CHECK( C > 0 ); }
	
	//! Constructor featuring two regularization parameters
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  negativeC   regularization parameter of the negative class (label 0)
	//! \param  positiveC    regularization parameter of the positive class (label 1)
	//! \param offset train svm with offset - this is not supported for all SVM solvers.
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	AbstractSvmTrainer(KernelType* kernel, double negativeC, double positiveC, bool offset, bool unconstrained = false)
	: m_kernel(kernel)
	, m_regularizers(2)
	, m_trainOffset(offset)
	, m_unconstrained(unconstrained)
	, m_cacheSize(0x4000000)
	{ 
		RANGE_CHECK( positiveC > 0 ); 
		RANGE_CHECK( negativeC > 0 ); 
		m_regularizers[0] = negativeC;
		m_regularizers[1] = positiveC;
		
	}

	/// \brief Return the value of the regularization parameter C.
	double C() const
	{
		SIZE_CHECK(m_regularizers.size() == 1);
		return m_regularizers[0];
	}
	
	RealVector const& regularizationParameters() const
	{
		return m_regularizers;
	}
	
	RealVector& regularizationParameters()
	{
		return m_regularizers;
	}
	
	KernelType* kernel()
	{ return m_kernel; }
	const KernelType* kernel() const
	{ return m_kernel; }
	void setKernel(KernelType* kernel)
	{ m_kernel = kernel; }

	bool isUnconstrained() const
	{ return m_unconstrained; }
	
	bool trainOffset() const
	{ return m_trainOffset; }

	double CacheSize() const
	{ return m_cacheSize; }
	void setCacheSize( std::size_t size )
	{ m_cacheSize = size; }

	/// get the hyper-parameter vector
	RealVector parameterVector() const
	{
		size_t kp = m_kernel->numberOfParameters();
		RealVector ret(kp + m_regularizers.size());
		if(m_unconstrained)
			init(ret) << parameters(m_kernel), log(m_regularizers);
		else
			init(ret) << parameters(m_kernel), m_regularizers;
		return ret;
	}

	/// set the vector of hyper-parameters
	void setParameterVector(RealVector const& newParameters)
	{
		size_t kp = m_kernel->numberOfParameters();
		SHARK_ASSERT(newParameters.size() == kp + m_regularizers.size());
		init(newParameters) >> parameters(m_kernel), m_regularizers;
		if(m_unconstrained)
			m_regularizers = exp(m_regularizers);
	}

	/// return the number of hyper-parameters
	size_t numberOfParameters() const{ 
		return m_kernel->numberOfParameters() + m_regularizers.size();
	}

protected:
	KernelType* m_kernel;               ///< Kernel object.
	///\brief Vector of regularization parameters. 
	///
	/// If the size of the vector is 1 there is only one regularization parameter for all classes, else there must
	/// be one for every class in the dataset.
	/// The exact meaning depends on the sub-class, but the value is always positive, 
	/// and higher implies a less regular solution.
	RealVector m_regularizers;
	bool m_trainOffset;
	bool m_unconstrained;               ///< Is log(C) stored internally as a parameter instead of C? If yes, then we get rid of the constraint C > 0 on the level of the parameter interface.
	std::size_t m_cacheSize;            ///< Number of values in the kernel cache. The size of the cache in bytes is the size of one entry (4 for float, 8 for double) times this number.
};


///
/// \brief Super class of all linear SVM trainers.
///
/// \par
/// This class is analogous to the AbstractSvmTrainer class,
/// but for training of linear SVMs. It represents the
/// regularization parameter of the SVM. The class also manages
/// meta-information of the training process, like the stopping
/// criterion and information on the actual solution.
///
template <class InputType>
class AbstractLinearSvmTrainer
: public AbstractTrainer<LinearClassifier<InputType>, unsigned int>
, public QpConfig
, public IParameterizable
{
public:
	typedef AbstractTrainer<LinearClassifier<InputType>, unsigned int> base_type;
	typedef LinearClassifier<InputType> ModelType;

	//! Constructor
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	AbstractLinearSvmTrainer(double C, bool unconstrained = false)
	: m_C(C)
	, m_unconstrained(unconstrained)
	{ RANGE_CHECK( C > 0 ); }

	/// \brief Return the value of the regularization parameter C.
	double C() const
	{ return m_C; }

	/// \brief Set the value of the regularization parameter C.
	void setC(double C) {
		RANGE_CHECK( C > 0 );
		m_C = C;
	}

	/// \brief Is the regularization parameter provided in logarithmic (unconstrained) form as a parameter?
	bool isUnconstrained() const
	{ return m_unconstrained; }

	/// \brief Get the hyper-parameter vector.
	RealVector parameterVector() const
	{
		RealVector ret(1);
		ret(0) = (m_unconstrained ? std::log(m_C) : m_C);
		return ret;
	}

	/// \brief Set the vector of hyper-parameters.
	void setParameterVector(RealVector const& newParameters)
	{
		SHARK_ASSERT(newParameters.size() == 1);
		setC(m_unconstrained ? std::exp(newParameters(0)) : newParameters(0));
	}

	/// \brief Return the number of hyper-parameters.
	size_t numberOfParameters() const
	{ return 1; }

	using QpConfig::m_stoppingcondition;
	using QpConfig::m_solutionproperties;
	using QpConfig::m_verbosity;

protected:
	double m_C;                         ///< Regularization parameter. The exact meaning depends on the sub-class, but the value is always positive, and higher implies a less regular solution.
	bool m_unconstrained;               ///< Is log(C) stored internally as a parameter instead of C? If yes, then we get rid of the constraint C > 0 on the level of the parameter interface.
};


}
#endif
