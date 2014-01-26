/*!
 * 
 *
 * \brief       Leave-one-out error for C-SVMs
 * 
 * 
 *
 * \author      T.Glasmachers
 * \date        2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_LOOERRORCSVM_H
#define SHARK_OBJECTIVEFUNCTIONS_LOOERRORCSVM_H


#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Models/Kernels/KernelExpansion.h>

namespace shark {

///
/// \brief Leave-one-out error, specifically optimized for C-SVMs.
///
template<class InputType, class CacheType = float>
class LooErrorCSvm : public SupervisedObjectiveFunction<InputType, unsigned int>
{
public:

	typedef CacheType QpFloatType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef LabeledData<InputType, unsigned int> DatasetType;
private:
	typedef SupervisedObjectiveFunction<InputType, unsigned int> base_type;

	const DatasetType* mep_dataset;
	KernelType* mep_kernel;
	bool m_withOffset;

public:
	/// \brief Constructor.
	LooErrorCSvm(DatasetType const& dataset, KernelType* kernel, bool withOffset)
	: mep_dataset(&dataset)
	, mep_kernel(kernel)
	, m_withOffset(withOffset)
	{
		SHARK_CHECK(kernel != NULL, "kernel is not allowed to be Null");
		base_type::m_features |= base_type::HAS_VALUE;
	}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LooErrorCSvm"; }

	/// inherited from SupervisedObjectiveFunction
	void setDataset(DatasetType const& dataset) {
		mep_dataset = &dataset;
	}
	
	std::size_t numberOfVariables()const{
		return mep_kernel->numberOfParameters()+1;
	}

	/// Evaluate the leave-one-out error for the given parameters. 
	/// These parameters describe the regularization 
	/// constant and the kernel parameters.
	double eval(const RealVector& params){
		QpStoppingCondition stop;
		return eval(params, stop);
	}
	/// Evaluate the leave-one-out error for the given parameters.
	/// These parameters describe the regularization constant and
	/// the kernel parameters.  Furthermore, the stopping
	/// conditions for the solver are passed as an argument.
	double eval(const RealVector& params, QpStoppingCondition &stop){
		this->m_evaluationCounter++;

		double C;
		blas::init(params)>>parameters(*mep_kernel),C;
		
		ZeroOneLoss<unsigned int, RealVector> loss;

		// prepare the quadratic program
		typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
		typedef CachedMatrix< KernelMatrixType > CachedMatrixType;
		typedef CSVMProblem<CachedMatrixType> SVMProblemType;
		KernelMatrixType km(*mep_kernel, mep_dataset->inputs());
		CachedMatrixType matrix(&km);
		SVMProblemType svmProblem(matrix,mep_dataset->labels(),C);
		std::size_t ell = km.size();
		
		//QpStoppingCondition stop;

		if (m_withOffset)
		{
			// solve the full problem with equality constraint and activated shrinking
			typedef SvmProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem);
			QpSolver< ProblemType > solver(problem);
			solver.solve(stop);
			RealVector alphaFull(problem.dimensions());
			for(std::size_t i = 0; i != problem.dimensions(); ++i){
				alphaFull(i) = problem.alpha(i);
			}
			KernelExpansion<InputType> svm(mep_kernel,mep_dataset->inputs(),true);

			// leave-one-out
			//problem.setShrinking(false);
			double mistakes = 0;
			for (std::size_t i=0; i<ell; i++)
			{
				// use sparseness of the solution:
				if (alphaFull(i) == 0.0) continue;
				problem.deactivateVariable(i);
				
				// solve the reduced problem
				solver.solve(stop);

				// predict using the previously removed example.
				// we need to take into account that the initial problem is solved
				// with shrinking and we thus need to get the initial permutation 
				// for the element index and the unpermuted alpha for the svm
				column(svm.alpha(),0)= problem.getUnpermutedAlpha();
				svm.offset(0) = computeBias(problem);
				std::size_t elementIndex = i;//svmProblem.permutation[i];
				unsigned int target = mep_dataset->element(elementIndex).label;
				mistakes += loss(target, svm(mep_dataset->element(elementIndex).input));
				
				problem.activateVariable(i);
			}
			return mistakes / (double)ell;
		}
		else
		{
			// solve the full problem without equality constraint and activated shrinking
			typedef BoxConstrainedProblem<SVMProblemType> ProblemType;
			ProblemType problem(svmProblem);
			QpSolver< ProblemType > solver(problem);
			solver.solve(stop);
			RealVector alphaFull(problem.dimensions());
			for(std::size_t i = 0; i != problem.dimensions(); ++i){
				alphaFull(i) = problem.alpha(i);
			}
			KernelExpansion<InputType> svm(mep_kernel,mep_dataset->inputs(),false);
			
			// leave-one-out
			//problem.setShrinking(false);
			double mistakes = 0;
			for (std::size_t i=0; i<ell; i++)
			{
				// use sparseness of the solution:
				if (alphaFull(i) == 0.0) continue;
				problem.deactivateVariable(i);
				

				// solve the reduced problem
				solver.solve(stop);

				// predict using the previously removed example.
				// we need to take into account that the initial problem is solved
				// with shrinking and we thus need to get the initial permutation 
				// for the element index and the unpermuted alpha for the svm
				column(svm.alpha(),0)= problem.getUnpermutedAlpha();
				std::size_t elementIndex = i;//svmProblem.permutation[i];
				unsigned int target = mep_dataset->element(elementIndex).label;
				mistakes += loss(target, svm(mep_dataset->element(elementIndex).input));

				problem.activateVariable(i);
			}
			return mistakes / (double)ell;
		}
	}

protected:
	/// Compute the SVM offset term (b).
	template<class Problem>
	double computeBias(Problem const& problem){
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		std::size_t ell = problem.dimensions();
		for (std::size_t i=0; i<ell; i++)
		{
			double value = problem.gradient(i);
			if (problem.alpha(i) == problem.boxMin(i))
			{
				lowerBound = std::max(value,lowerBound);
			}
			else if (problem.alpha(i) == problem.boxMax(i))
			{
				upperBound = std::min(value,upperBound);
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}
		if (freeVars > 0) 
			return sum / freeVars;
		else 
			return 0.5 * (lowerBound + upperBound);
	}
};


}
#endif
