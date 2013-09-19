//===========================================================================
/*!
 *  \brief Trainer for the Epsilon-Support Vector Machine for Regression
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


#ifndef SHARK_ALGORITHMS_EPSILONSVMTRAINER_H
#define SHARK_ALGORITHMS_EPSILONSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/SvmProblems.h>

namespace shark {


///
/// \brief Training of Epsilon-SVMs for regression.
///
/// The Epsilon-SVM is a support vector machine variant
/// for regression problems. Given are data tuples
/// \f$ (x_i, y_i) \f$ with x-component denoting input and
/// y-component denoting a real-valued label (see the tutorial on
/// label conventions; the implementation uses RealVector),
/// a kernel function k(x, x'), a regularization constant C > 0,
/// and a loss insensitivity parameter \f$ \varepsilon \f$.
/// Let H denote the kernel induced reproducing kernel Hilbert
/// space of k, and let \f$ \phi \f$ denote the corresponding
/// feature map. Then the SVM regression function is of the form
/// \f[
///     (x) = \langle w, \phi(x) \rangle + b
/// \f]
/// with coefficients w and b given by the (primal)
/// optimization problem
/// \f[
///     \min \frac{1}{2} \|w\|^2 + C \sum_i L(y_i, f(x_i)),
/// \f]
/// where
/// \f[
///     L(y, f(x)) = \max\{0, |y - f(x)| - \varepsilon \}
/// \f]
/// is the \f$ \varepsilon \f$ insensitive absolute loss.
///
template <class InputType, class CacheType = float>
class EpsilonSvmTrainer : public AbstractSvmTrainer<InputType, RealVector, KernelExpansion<InputType> >
{
public:

	typedef CacheType QpFloatType;

	typedef KernelMatrix< InputType, QpFloatType > KernelMatrixType;
	typedef BlockMatrix2x2< KernelMatrixType > BlockMatrixType;
	typedef CachedMatrix< BlockMatrixType > CachedBlockMatrixType;
	typedef PrecomputedMatrix< BlockMatrixType > PrecomputedBlockMatrixType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, RealVector, KernelExpansion<InputType> > base_type;

	/// Constructor
	/// \param  kernel         kernel function to use for training and prediction
	/// \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	/// \param  epsilon        Loss insensitivity parameter.
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	EpsilonSvmTrainer(KernelType* kernel, double C, double epsilon, bool unconstrained = false)
	: base_type(kernel, C, true, unconstrained)
	, m_epsilon(epsilon)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "EpsilonSvmTrainer"; }

	double epsilon() const
	{ return m_epsilon; }
	void setEpsilon(double epsilon)
	{ m_epsilon = epsilon; }

	/// get the hyper-parameter vector
	RealVector parameterVector() const
	{
		size_t sp = base_type::numberOfParameters();
		RealVector ret(sp + 1);
		blas::init(ret)<< base_type::parameterVector(),(base_type::m_unconstrained ? std::log(m_epsilon) : m_epsilon);
		return ret;
	}

	/// set the vector of hyper-parameters
	void setParameterVector(RealVector const& newParameters)
	{
		size_t sp = base_type::numberOfParameters();
		SHARK_ASSERT(newParameters.size() == sp + 1);
		base_type::setParameterVector(subrange(newParameters, 0, sp));
		setEpsilon(base_type::m_unconstrained ? std::exp(newParameters(sp)) : newParameters(sp));
	}

	/// return the number of hyper-parameters
	size_t numberOfParameters() const
	{ return (base_type::numberOfParameters() + 1); }

	void train(KernelExpansion<InputType>& svm, LabeledData<InputType, RealVector> const& dataset){
		svm.setStructure(base_type::m_kernel,dataset.inputs(),true,1);
		
		SHARK_CHECK(labelDimension(dataset) == 1, "[EpsilonSvmTrainer::train] can only train 1D labels");

		if (QpConfig::precomputeKernel())
			trainSVM<PrecomputedBlockMatrixType>(svm,dataset);
		else
			trainSVM<CachedBlockMatrixType>(svm,dataset);
		
		if (base_type::sparsify()) svm.sparsify();
	}

private:
	template<class MatrixType>
	void trainSVM(KernelExpansion<InputType>& svm, LabeledData<InputType, RealVector> const& dataset){
		typedef GeneralQuadraticProblem<MatrixType> SVMProblemType;
		typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
		
		//Set up the problem
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());
		std::size_t ic = km.size();
		BlockMatrixType blockkm(&km);
		MatrixType matrix(&blockkm);
		SVMProblemType svmProblem(matrix);
		for(std::size_t i = 0; i != ic; ++i){
			svmProblem.linear(i) = dataset.element(i).label(0) - m_epsilon;
			svmProblem.linear(i+ic) = dataset.element(i).label(0) + m_epsilon;
			svmProblem.boxMin(i) = 0;
			svmProblem.boxMax(i) = this->C();
			svmProblem.boxMin(i+ic) = -this->C();
			svmProblem.boxMax(i+ic) = 0;
		}
		ProblemType problem(svmProblem,base_type::m_shrinking);
		
		//solve it
		QpSolver< ProblemType> solver(problem);
		solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
		RealVector alpha = problem.getUnpermutedAlpha();
		column(svm.alpha(),0)= subrange(alpha,0,ic)+subrange(alpha,ic,2*ic);
		
		// compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		
		std::size_t freeVars = 0;
		for (std::size_t i=0; i< ic; i++)
		{
			if (problem.alpha(i) > 0.0)
			{
				double value = problem.gradient(i);
				if (problem.alpha(i) < this->C())
				{
					sum += value;
					freeVars++;
				}
				else
				{
					lowerBound = std::max(value,lowerBound);
				}
			}
			if (problem.alpha(i + ic) < 0.0)
			{
				double value = problem.gradient(i + ic);
				if (problem.alpha(i + ic) > -this->C())
				{
					sum += value;
					freeVars++;
				}
				else
				{
					upperBound = std::min(value,upperBound);
				}
			}
		}
		if (freeVars > 0) 
			svm.offset(0) = sum / freeVars;		// stabilized (averaged) exact value
		else 
			svm.offset(0) = 0.5 * (lowerBound + upperBound);	// best estimate
		
		base_type::m_accessCount = km.getAccessCount();
	}
	double m_epsilon;
};


}
#endif
