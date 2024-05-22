/*!
 * 
 *
 * \brief       Radius Margin Quotient for SVM model selection
 * 
 * 
 *
 * \author      T.Glasmachers, O.Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_RADIUSMARGINQUOTIENT_H
#define SHARK_OBJECTIVEFUNCTIONS_RADIUSMARGINQUOTIENT_H


#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Models/Kernels/KernelHelpers.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/KernelMatrix.h>

namespace shark {


///
/// \brief radius margin quotions for binary SVMs
///
/// \par
/// The RadiusMarginQuotient is the quotient \f$ R^2 / \rho^2 \f$
/// of the radius R of the smallest sphere containing the
/// training data and the margin \f$\rho\f$ of a binary hard margin
/// support vector machine. Both distances depend on the
/// kernel function, and thus on its parameters.
/// The radius margin quotient is a common objective
/// function for the adaptation of the kernel parameters
/// of a binary hard-margin SVM.
/// \ingroup kerneloptimization
template<class InputType, class CacheType = float>
class RadiusMarginQuotient : public AbstractObjectiveFunction< RealVector, double >
{
public:
	typedef CacheType QpFloatType;

	typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;

	typedef LabeledData<InputType, unsigned int> DatasetType;
	typedef AbstractKernelFunction<InputType> KernelType;

	/// \brief Constructor.
	RadiusMarginQuotient(DatasetType const& dataset, KernelType* kernel)
	: mep_kernel(kernel),m_dataset(dataset)
	{
		m_features |= HAS_VALUE;
		if (mep_kernel->hasFirstParameterDerivative())
			m_features |= HAS_FIRST_DERIVATIVE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RadiusMarginQuotient"; }

	std::size_t numberOfVariables()const{
		return mep_kernel->numberOfParameters();
	}

	/// \brief Evaluate the radius margin quotient.
	///
	/// \par
	/// The parameters are passed into the kernel, and the
	/// radius-margin quotient is computed w.r.t. the
	/// kernel-induced metric.
	double eval(SearchPointType const& parameters) const{
		SIZE_CHECK(parameters.size() == mep_kernel->numberOfParameters());
		SHARK_RUNTIME_CHECK(! m_dataset.empty(), "[RadiusMarginQuotient::eval] call setDataset first");
		m_evaluationCounter++;
		
		
		mep_kernel->setParameterVector(parameters);

		Result result = computeRadiusMargin();

		return result.w2 * result.R2;
	}

	/// \brief Evaluate the radius margin quotient and its first derivative.
	///
	/// \par
	/// The parameters are passed into the kernel, and the
	/// radius-margin quotient and its derivative are computed
	/// w.r.t. the kernel-induced metric.
	double evalDerivative(SearchPointType const& parameters, FirstOrderDerivative& derivative) const{
		SHARK_RUNTIME_CHECK(! m_dataset.empty(), "[RadiusMarginQuotient::evalDerivative] call setDataset first");
		SIZE_CHECK(parameters.size() == mep_kernel->numberOfParameters());
		m_evaluationCounter++;
		
		mep_kernel->setParameterVector(parameters);

		Result result = computeRadiusMargin();
		
		derivative = calculateKernelMatrixParameterDerivative(
			*mep_kernel, m_dataset.inputs(),
			result.w2*(to_diagonal(result.beta)-outer_prod(result.beta,result.beta))
			-result.R2*outer_prod(result.alpha,result.alpha)
		);
		
		
		return result.w2 * result.R2;
	}

protected:
	struct Result{
		RealVector alpha;
		RealVector beta;
		double w2;
		double R2;
	};
	
	Result computeRadiusMargin()const{
		std::size_t ell = m_dataset.numberOfElements();
		
		QpStoppingCondition stop;
		Result result;
		{
			KernelMatrixType km(*mep_kernel, m_dataset.inputs());
			CachedMatrixType cache(&km);
			typedef CSVMProblem<CachedMatrixType> SVMProblemType;
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			
			SVMProblemType svmProblem(cache,m_dataset.labels(),1e100);
			ProblemType problem(svmProblem);
			
			QpSolver< ProblemType> solver(problem);
			QpSolutionProperties prop;
			solver.solve(stop, &prop);
			result.w2 = 2.0 * prop.value;
			result.alpha = problem.getUnpermutedAlpha();
		}
		{
			// create and solve the radius problem (also a quadratic program)
			KernelMatrixType km(*mep_kernel, m_dataset.inputs());
			CachedMatrixType cache(&km);
			typedef BoxedSVMProblem<CachedMatrixType> SVMProblemType;
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			
			// Setup the problem
			RealVector linear(ell);
			for (std::size_t i=0; i<ell; i++){
				linear(i) = 0.5 * km(i, i);
			}
			SVMProblemType svmProblem(cache,linear,0.0,1.0);
			ProblemType problem(svmProblem);
			
			//solve it
			QpSolver< ProblemType> solver(problem);
			QpSolutionProperties prop;
			solver.solve(stop, &prop);
			result.R2 = 2.0 * prop.value;
			result.beta = problem.getUnpermutedAlpha();
		}
		return result;
	}
	
	KernelType* mep_kernel;            ///< underlying parameterized kernel object
	DatasetType m_dataset;                  ///< labeled data for radius and (hard) margin computation
	
};


}
#endif
