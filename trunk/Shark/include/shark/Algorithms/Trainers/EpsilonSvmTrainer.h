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
class EpsilonSvmTrainer : public AbstractSvmTrainer<InputType, RealVector>
{
public:

	/// \brief Convenience typedefs:
	/// this and many of the below typedefs build on the class template type CacheType.
	/// Simply changing that one template parameter CacheType thus allows to flexibly
	/// switch between using float or double as type for caching the kernel values.
	/// The default is float, offering sufficient accuracy in the vast majority
	/// of cases, at a memory cost of only four bytes. However, the template
	/// parameter makes it easy to use double instead, (e.g., in case high
	/// accuracy training is needed).
	typedef CacheType QpFloatType;
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

	typedef KernelMatrix< InputType, QpFloatType > KernelMatrixType;
	typedef BlockMatrix2x2< KernelMatrixType > BlockMatrixType;
	typedef CachedMatrix< BlockMatrixType > CachedBlockMatrixType;
	typedef PrecomputedMatrix< BlockMatrixType > PrecomputedBlockMatrixType;

	typedef AbstractModel<InputType, RealVector> ModelType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, RealVector> base_type;

	/// Constructor
	/// \param  kernel         kernel function to use for training and prediction
	/// \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	/// \param  epsilon        Loss insensitivity parameter.
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	EpsilonSvmTrainer(KernelType* kernel, double C, double epsilon, bool unconstrained = false)
	: base_type(kernel, C, unconstrained)
	, m_epsilon(epsilon)
	{
		base_type::m_name = "EpsilonSvmTrainer";
	}

	double epsilon() const
	{ return m_epsilon; }
	void setEpsilon(double epsilon)
	{ m_epsilon = epsilon; }

	/// get the hyper-parameter vector
	RealVector parameterVector() const
	{
		size_t sp = base_type::numberOfParameters();
		RealVector ret(sp + 1);
		RealVectorRange(ret, Range(0, sp)) = base_type::parameterVector();
		ret(sp) = (base_type::m_unconstrained ? std::log(m_epsilon) : m_epsilon);
		return ret;
	}

	/// set the vector of hyper-parameters
	void setParameterVector(RealVector const& newParameters)
	{
		size_t sp = base_type::numberOfParameters();
		SHARK_ASSERT(newParameters.size() == sp + 1);
		base_type::setParameterVector(ConstRealVectorRange(newParameters, Range(0, sp)));
		setEpsilon(base_type::m_unconstrained ? std::exp(newParameters(sp)) : newParameters(sp));
	}

	/// return the number of hyper-parameters
	size_t numberOfParameters() const
	{ return (base_type::numberOfParameters() + 1); }

	void train(KernelExpansion<InputType>& svm, const LabeledData<InputType, RealVector>& dataset)
	{
		// Setup the cached kernel matrix
		KernelMatrixType km(*base_type::m_kernel, dataset.inputs());
		BlockMatrixType km2(&km);

		SHARK_CHECK(svm.hasOffset(), "[EpsilonSvmTrainer::train] training of models without offset is not supported");
		SHARK_CHECK(svm.outputSize() == 1, "[EpsilonSvmTrainer::train] wrong number of outputs in the kernel expansion");

		// prepare the quadratic program description
		std::size_t i, ic = dataset.numberOfElements();
		RealVector linear(2*ic);
		RealVector lower(2*ic);
		RealVector upper(2*ic);
		RealVector alpha(2*ic,0.0);
		for (i=0; i<ic; i++)
		{
// 			double a = param(i);
// 			if (a > 0.0)
// 			{
// 				alpha(i) = a;
// 				alpha(i + ic) = 0.0;
// 			}
// 			else
// 			{
// 				alpha(i) = 0.0;
// 				alpha(i + ic) = -a;
// 			}
			linear(i) = dataset.element(i).label(0) - m_epsilon;
			lower(i) = 0.0;
			upper(i) = base_type::m_C;
			linear(i + ic) = dataset.element(i).label(0) + m_epsilon;
			lower(i + ic) = -base_type::m_C;
			upper(i + ic) = 0.0;
		}

		// solve the quadratic program
		RealVector gradient;
		if (base_type::precomputeKernel())
		{
			PrecomputedBlockMatrixType matrix(&km2);
			QpSvmDecomp< PrecomputedBlockMatrixType > solver(matrix);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			gradient = solver.getGradient();
		}
		else
		{
			CachedBlockMatrixType matrix(&km2, base_type::m_cacheSize );
			QpSvmDecomp< CachedBlockMatrixType > solver(matrix);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			gradient = solver.getGradient();
		}

		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());

		RealVector param(ic + 1);
		for (i=0; i<ic; i++) param(i) = alpha(i) + alpha(i + ic);

		// compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		for (i=0; i<ic; i++)
		{
			if (alpha(i) > 0.0)
			{
				double value = gradient(i);
				if (alpha(i) < base_type::m_C)
				{
					sum += value;
					freeVars++;
				}
				else
				{
					if (value > lowerBound) lowerBound = value;
				}
			}
			if (alpha(i + ic) < 0.0)
			{
				double value = gradient(i + ic);
				if (alpha(i + ic) > -base_type::m_C)
				{
					sum += value;
					freeVars++;
				}
				else
				{
					if (value < upperBound) upperBound = value;
				}
			}
		}
		if (freeVars > 0) param(ic) = sum / freeVars;		// stabilized (averaged) exact value
		else param(ic) = 0.5 * (lowerBound + upperBound);	// best estimate

		// write the solution into the model
		svm.setParameterVector(param);

		base_type::m_accessCount = km.getAccessCount();
		if (base_type::sparsify()) svm.sparsify();
	}

protected:
	double m_epsilon;
};


}
#endif
