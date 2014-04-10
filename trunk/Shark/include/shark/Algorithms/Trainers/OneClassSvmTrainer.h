//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for One-Class Support Vector Machines
 * 
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


#ifndef SHARK_ALGORITHMS_ONECLASSSVMTRAINER_H
#define SHARK_ALGORITHMS_ONECLASSSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/KernelMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>


namespace shark {


///
/// \brief Training of one-class SVMs.
///
/// The one-class support vector machine is an unsupervised
/// method for learning the high probability region of a
/// distribution. Given are data points \f$ x_i, i \in \{1, \dots, m\} \f$,
/// a kernel function k(x, x') and a regularization
/// constant C > 0. Let H denote the kernel induced
/// reproducing kernel Hilbert space of k, and let \f$ \phi \f$
/// denote the corresponding feature map.
/// Then an estimate of a high probability region of the
/// distribution generating the sample points is described
/// by the set where the following function takes positive
/// values:
/// \f[
///     f(x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + \frac{1}{\nu m} \sum_{i=1}^m \xi_i - \rho
/// \f]
/// \f[
///     \text{s.t. } \langle w, \phi(x_i) \rangle + b \geq \rho - \xi_i; \xi_i \geq 0
/// \f]
/// \f$ 0 \leq \nu, \rho \leq 1 \f$ are parameters of the
/// method for controlling the smoothness of the solution
/// and the amount of probability mass covered.
///
/// For more details refer to the paper:<br/>
/// <p>Estimating the support of a high-dimensional distribution. B. Sch&ouml;lkopf, J. C. Platt, J. Shawe-Taylor, A. Smola, and R. C. Williamson, 1999.</p>
///
template <class InputType, class CacheType = float>
class OneClassSvmTrainer : public AbstractUnsupervisedTrainer<KernelExpansion<InputType> >, public QpConfig, public IParameterizable
{
public:

	typedef CacheType QpFloatType;
	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef QpConfig base_type;

	typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
	typedef PrecomputedMatrix< KernelMatrixType > PrecomputedMatrixType;

	OneClassSvmTrainer(KernelType* kernel, double nu)
	: m_kernel(kernel)
	, m_nu(nu)
	, m_cacheSize(0x4000000)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "OneClassSvmTrainer"; }

	double nu() const
	{ return m_nu; }
	void setNu(double nu)
	{ m_nu = nu; }
	KernelType* kernel()
	{ return m_kernel; }
	const KernelType* kernel() const
	{ return m_kernel; }
	void setKernel(KernelType* kernel)
	{ m_kernel = kernel; }

	double CacheSize() const
	{ return m_cacheSize; }
	void setCacheSize( std::size_t size )
	{ m_cacheSize = size; }

	/// get the hyper-parameter vector
	RealVector parameterVector() const
	{
		size_t kp = m_kernel->numberOfParameters();
		RealVector ret(kp + 1);
		RealVectorRange(ret, Range(0, kp)) = m_kernel->parameterVector();
		ret(kp) = m_nu;
		return ret;
	}

	/// set the vector of hyper-parameters
	void setParameterVector(RealVector const& newParameters)
	{
		size_t kp = m_kernel->numberOfParameters();
		SHARK_ASSERT(newParameters.size() == kp + 1);
		m_kernel->setParameterVector(ConstRealVectorRange(newParameters, Range(0, kp)));
		setNu(newParameters(kp));
	}

	/// return the number of hyper-parameters
	size_t numberOfParameters() const
	{ return (m_kernel->numberOfParameters() + 1); }

	void train(KernelExpansion<InputType>& svm, UnlabeledData<InputType> const& inputset)
	{
		SHARK_CHECK(m_nu > 0.0 && m_nu< 1.0, "[OneClassSvmTrainer::train] invalid setting of the parameter nu (must be 0 < nu < 1)");
		svm.setStructure(m_kernel,inputset,true);

		// solve the quadratic program
		if (QpConfig::precomputeKernel())
			trainSVM<PrecomputedMatrixType>(svm,inputset);
		else
			trainSVM<CachedMatrixType>(svm,inputset);

		if (base_type::sparsify()) 
			svm.sparsify();
	}

protected:
	KernelType* m_kernel;
	double m_nu;
	std::size_t m_cacheSize;

	template<class MatrixType>
	void trainSVM(KernelExpansion<InputType>& svm, UnlabeledData<InputType> const& inputset){
		typedef BoxedSVMProblem<MatrixType> SVMProblemType;
		typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
		
		// Setup the problem
		
		KernelMatrixType km(*m_kernel, inputset);
		MatrixType matrix(&km);
		std::size_t ic = matrix.size();
		double upper = 1.0/(m_nu*ic);
		SVMProblemType svmProblem(matrix,blas::repeat(0.0,ic),0.0,upper);
		ProblemType problem(svmProblem,base_type::m_shrinking);
		
		//solve it
		QpSolver< ProblemType > solver(problem);
		solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
		column(svm.alpha(),0)= problem.getUnpermutedAlpha();
		
		// compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		for (std::size_t i=0; i != problem.dimensions(); i++)
		{
			double value = problem.gradient(i);
			if (problem.alpha(i) == 0.0)
				lowerBound = std::max(value,lowerBound);
			else if (problem.alpha(i) == upper)
				upperBound = std::min(value,upperBound);
			else
			{
				sum += value;
				freeVars++;
			}
		}
		if (freeVars > 0)
			svm.offset(0) = sum / freeVars;		// stabilized (averaged) exact value
		else 
			svm.offset(0) = 0.5 * (lowerBound + upperBound);	// best estimate
		
		base_type::m_accessCount = km.getAccessCount();
	}
};


}
#endif
