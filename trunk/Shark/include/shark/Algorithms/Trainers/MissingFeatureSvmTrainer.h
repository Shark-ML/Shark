//===========================================================================
/*!
 * 
 *
 * \brief       Trainer for binary SVMs natively supporting missing features.
 * 
 * 
 *
 * \author      B. Li
 * \date        2012
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

#ifndef SHARK_ALGORITHMS_TRAINERS_MISSING_FEATURE_SVM_H
#define SHARK_ALGORITHMS_TRAINERS_MISSING_FEATURE_SVM_H

#include "shark/Algorithms/Trainers/AbstractSvmTrainer.h"
#include "shark/Models/Kernels/EvalSkipMissingFeatures.h"
#include "shark/Models/Kernels/KernelExpansion.h"
#include "shark/Models/Kernels/MissingFeaturesKernelExpansion.h"
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/ExampleModifiedKernelMatrix.h>

#include <boost/foreach.hpp>

namespace shark {

/// \brief Trainer for binary SVMs natively supporting missing features.
///
/// This is a specialized variant of the standard binary C-SVM which can be used
/// to learn from data with missing features, without the need for prior imputation
/// of missing values. The key idea is that each example is considered as having an
/// instance-specific margin value, which is computed in the lower-dimensional subspace
/// for which all features of that example are present.
///
/// The resulting optimization problem has the drawback that it is not convex any
/// more. However, a local minimum can be easily reached by an iterative wrapper
/// algorithm around a standard C-SVM solver. In detail, example-wise weights \f$ s_i \f$
/// are incorporated into the quadratic term of the standard SVM optimization problem.
/// These take initial values of 1, and are then iteratively updated according to the
/// instance-specific margin values. After each such update, the SVM solver is again called,
/// and so on. Usually, between 2 and 5 iterations have been shown to be sufficient for
/// an acceptable level of convergence.
///
/// For details, see the paper:<br/>
/// <p>Max-margin Classification of %Data with Absent Features
/// Gal Chechik, Geremy Heitz, Gal Elidan, Pieter Abbeel and Daphne Koller. JMLR 2008.</p>
///
/// \note Much of the code in this class is borrowed from the class CSvmTrainer
template <class InputType, class CacheType = float>
class MissingFeatureSvmTrainer : public AbstractSvmTrainer<InputType, unsigned int,MissingFeaturesKernelExpansion<InputType> >
{
protected:

	typedef CacheType QpFloatType;
	typedef AbstractKernelFunction<InputType> KernelType;
	typedef AbstractSvmTrainer<InputType, unsigned int,MissingFeaturesKernelExpansion<InputType> > base_type;

public:

	/// Constructor
	MissingFeatureSvmTrainer(KernelType* kernel, double C, bool offset, bool unconstrained = false)
	:
		base_type(kernel, C, offset, unconstrained),
		m_maxIterations(4u)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MissingFeatureSvmTrainer"; }

	void train(MissingFeaturesKernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		// Check prerequisites
		SHARK_CHECK(numberOfClasses(dataset) == 2, "[MissingFeatureSvmTrainer::train] Not a binary problem");

		svm.setStructure(base_type::m_kernel,dataset.inputs(), this->m_trainOffset);
		
		if(svm.hasOffset())
			trainWithOffset(svm,dataset);
		else
			trainWithoutOffset(svm,dataset);
	}

	void setMaxIterations(std::size_t newIterations)
	{ m_maxIterations = newIterations; }

private:

	/// Number of iterations to run
	std::size_t m_maxIterations;

	void trainWithoutOffset(MissingFeaturesKernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		
		// Initialize scaling coefficients as 1.0
		RealVector scalingCoefficients(dataset.numberOfElements(), 1.0);

		// What body of this loop does:
		//   *) Solve the QP with a normal solver treating s_i as constants
		//   *) Calculate norms: w_i and w
		//   *) Update s_i with w_i / w
		for(std::size_t iteration = 0; iteration != m_maxIterations; ++iteration){
			//Set up the problem
			typedef ExampleModifiedKernelMatrix<InputType, QpFloatType> MatrixType;
			typedef CachedMatrix<MatrixType> CachedMatrixType;
			typedef CSVMProblem<CachedMatrixType> SVMProblemType;
			typedef BoxConstrainedShrinkingProblem<SVMProblemType> ProblemType;
			MatrixType kernelMatrix(*base_type::m_kernel, dataset.inputs());
			kernelMatrix.setScalingCoefficients(scalingCoefficients);
			CachedMatrixType matrix(&kernelMatrix);
			SVMProblemType svmProblem(matrix,dataset.labels(),this->C());
			ProblemType problem(svmProblem,base_type::m_shrinking);
			
			//solve it
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			RealVector alpha = problem.getUnpermutedAlpha();
			
			//update s_i and w_i
			const double classifierNorm = svm.computeNorm(alpha, scalingCoefficients);
			SHARK_ASSERT(classifierNorm > 0.0);
			for (std::size_t i = 0; i < scalingCoefficients.size(); ++i)
			{
				// Update scaling coefficients
				scalingCoefficients(i) = svm.computeNorm(
					alpha,
					scalingCoefficients,
					dataset.element(i).input)
					/ classifierNorm;
			}
			
			//store alpha in the last iteration inside the svm
			if(iteration == m_maxIterations-1)
				column(svm.alpha(),0)= alpha;

			//keep track of number of kernel evaluations
			base_type::m_accessCount += kernelMatrix.getAccessCount();
		}
		svm.setScalingCoefficients(scalingCoefficients);
	}
	void trainWithOffset(MissingFeaturesKernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		// Initialize scaling coefficients as 1.0
		std::size_t datasetSize = dataset.numberOfElements();
		RealVector scalingCoefficients(datasetSize, 1.0);

		// What body of this loop does:
		//   *) Solve the QP with a normal solver treating s_i as constants
		//   *) Calculate norms: w_i and w
		//   *) Update s_i with w_i / w
		for(std::size_t iteration = 0; iteration != m_maxIterations; ++iteration){
			//Set up the problem
			typedef ExampleModifiedKernelMatrix<InputType, QpFloatType> MatrixType;
			typedef CachedMatrix<MatrixType> CachedMatrixType;
			typedef CSVMProblem<CachedMatrixType> SVMProblemType;
			typedef SvmShrinkingProblem<SVMProblemType> ProblemType;
			MatrixType kernelMatrix(*base_type::m_kernel, dataset.inputs());
			kernelMatrix.setScalingCoefficients(scalingCoefficients);
			CachedMatrixType matrix(&kernelMatrix);
			SVMProblemType svmProblem(matrix,dataset.labels(),this->C());
			ProblemType problem(svmProblem,base_type::m_shrinking);
			
			//solve it
			QpSolver< ProblemType > solver(problem);
			solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
			RealVector unpermutedAlpha = problem.getUnpermutedAlpha();
			
			//update s_i and w_i
			const double classifierNorm = svm.computeNorm(unpermutedAlpha, scalingCoefficients);
			SHARK_ASSERT(classifierNorm > 0.0);
			for (std::size_t i = 0; i < scalingCoefficients.size(); ++i)
			{
				// Update scaling coefficients
				scalingCoefficients(i) = svm.computeNorm(
					unpermutedAlpha,
					scalingCoefficients,
					dataset.element(i).input
				)/ classifierNorm;
			}
			
			
			if(iteration == m_maxIterations-1){
				//in the last tieration,y
				// Compute the offset(i.e., b or Bias) and push it along with alpha to SVM
				column(svm.alpha(),0)= unpermutedAlpha;
				double lowerBound = -1e100;
				double upperBound = 1e100;
				double sum = 0.0;
				std::size_t freeVars = 0;

				// No reason to init to 0, but avoid compiler warnings
				for (std::size_t i = 0; i < datasetSize; i++)
				{
					// In case of no free SVs, we are looking for the largest gradient of all alphas at the lower bound
					// and the smallest gradient of all alphas at the upper bound
					const double value = problem.gradient(i);
					if (problem.alpha(i) == problem.boxMin(i)){
						lowerBound = std::max(value,lowerBound);
					}
					else if (problem.alpha(i) == problem.boxMax(i)){
						upperBound = std::min(value,upperBound);
					}
					else{
						sum += value;
						freeVars++;
					}
				}

				if (freeVars > 0) {
					// Stabilized (averaged) exact value
					svm.offset(0) = sum / freeVars;
				} else {
					// TODO: need work out how to do the calculation of the derivative with missing features

					// Return best estimate
					svm.offset(0) = 0.5 * (lowerBound + upperBound);
				}
			}

			//keep track of number of kernel evaluations
			base_type::m_accessCount += kernelMatrix.getAccessCount();
		}
		svm.setScalingCoefficients(scalingCoefficients);
	}

};

} // namespace shark {

#endif
