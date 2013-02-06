//===========================================================================
/*!
 *  \brief Trainer for binary SVMs natively supporting missing features.
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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

#ifndef SHARK_ALGORITHMS_TRAINERS_MISSING_FEATURE_SVM_H
#define SHARK_ALGORITHMS_TRAINERS_MISSING_FEATURE_SVM_H

#include "shark/Algorithms/QP/QuadraticProgram.h"
#include "shark/Algorithms/Trainers/AbstractSvmTrainer.h"
#include "shark/Models/Kernels/EvalSkipMissingFeatures.h"
#include "shark/Models/Kernels/KernelExpansion.h"
#include "shark/Models/Kernels/MissingFeaturesKernelExpansion.h"

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
	MissingFeatureSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	:
		base_type(kernel, C, unconstrained),
		m_maxIterations(4u)
	{
		base_type::m_name = "MissingFeatureSvmTrainer";
	}

	void train(MissingFeaturesKernelExpansion<InputType>& svm, LabeledData<InputType, unsigned int> const& dataset)
	{
		// Check prerequisites
		SHARK_CHECK(svm.outputSize() == 1, "[MissingFeatureSvmTrainer::train] wrong number of outputs in the kernel expansion");
		const std::size_t datasetSize = dataset.numberOfElements();

		// Set kernel & basis
		svm.setKernel(base_type::m_kernel);
		svm.setBasis(dataset.inputs());

		RealVector lower;
		RealVector upper;
		RealVector alpha;
		RealVector gradient;

		// Initialize scaling coefficients as 1.0
		RealVector scalingCoefficients(datasetSize, 1.0);

		// What body of this loop does:
		//   *) Solve the QP with a normal solver treating s_i as constants
		//   *) Calculate norms: w_i and w
		//   *) Update s_i with w_i / w
		std::size_t iterationCount = m_maxIterations;
		while ((iterationCount--) > 0)
		{
			lower = RealVector(datasetSize);
			upper = RealVector(datasetSize);
			alpha = RealZeroVector(datasetSize); // should be initialized to zero
			gradient = RealVector(datasetSize);

			computeAlpha(alpha, gradient, lower, upper, dataset, scalingCoefficients);
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
		}

		if (svm.hasOffset())
		{
			// Compute the offset(i.e., b or Bias) and push it along with alpha to SVM
			const double b = computeBiasAndSetItsDerivate(alpha, gradient, lower, upper, dataset);
			RealVector parameter(datasetSize + 1);
			RealVectorRange(parameter, Range(0, datasetSize)) = alpha;
			parameter(datasetSize) = b;
			svm.setParameterVector(parameter);
		}
		else
		{
			resetDerivateOfBias();
			svm.setParameterVector(alpha);
		}

		// Set scaling coefficients
		svm.setScalingCoefficients(scalingCoefficients);
	}

	void setMaxIterations(std::size_t newIterations)
	{ m_maxIterations = newIterations; }

	/// for the rare case that there are only bounded SVs and no free SVs,
	/// this gives access to the derivative of b w.r.t. C for external use
	RealVector& get_db_dParams() const
	{ throw SHARKEXCEPTION("[MissingFeatureSvmTrainer::get_db_dParams] not supported."); }

protected:

	/// In the rare case that there are only bounded SVs and no free SVs,
	/// so this is used to hold the derivative of b w.r.t. the hyperparameters
	RealVector m_db_dParams;

	/// Number of iterations to run
	std::size_t m_maxIterations;

private:

	/// Compute alpha and friends
	/// @param alpha[in, out] the alpha. Should be initialized to zero.
	/// @param gradient[out] the gradient
	/// @param lower[out] lower boundary
	/// @param upper[out] upper boundary
	/// @param dataset the dataset used to solve the problem
	/// @param scalingCoefficients the current scaling coefficients
	void computeAlpha(
		RealVector& alpha,
		RealVector& gradient,
		RealVector& lower,
		RealVector& upper,
		const LabeledData<InputType, unsigned int>& dataset,
		const RealVector& scalingCoeffecients)
	{
		const std::size_t datasetSize = dataset.numberOfElements();
		SIZE_CHECK(datasetSize > 0);

		// Prepare the quadratic program description
		RealVector linear(datasetSize);
		for (std::size_t i = 0; i < datasetSize; ++i)
		{
			if (0 == dataset.element(i).label)
			{
				linear(i) = -1.0;
				lower(i) = -base_type::m_C;
				upper(i) = 0.0;
			}
			else
			{
				SHARK_ASSERT(1 == dataset.element(i).label);
				linear(i) = 1.0;
				lower(i) = 0.0;
				upper(i) = base_type::m_C;
			}
		}

		// Create kernel matrix
		SHARK_ASSERT(base_type::m_kernel);
		ExampleModifiedKernelMatrix<InputType, QpFloatType> kernelMatrix(*base_type::m_kernel, dataset.inputs());
		kernelMatrix.setScalingCoefficients(scalingCoeffecients);

		// Solve the problem with equality constraint
		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrix<ExampleModifiedKernelMatrix<InputType, QpFloatType> > matrix(&kernelMatrix);
			QpSvmDecomp<PrecomputedMatrix< ExampleModifiedKernelMatrix<InputType, QpFloatType> > > solver(matrix);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			gradient = solver.getGradient();
		}
		else
		{
			CachedMatrix<ExampleModifiedKernelMatrix<InputType, QpFloatType> > matrix(&kernelMatrix);
			QpSvmDecomp<CachedMatrix<ExampleModifiedKernelMatrix<InputType, QpFloatType> > > solver(matrix);
			QpSolutionProperties& prop = base_type::m_solutionproperties;
			solver.setShrinking(base_type::m_shrinking);
			solver.solve(linear, lower, upper, alpha, base_type::m_stoppingcondition, &prop);
			gradient = solver.getGradient();
		}
		base_type::m_accessCount += kernelMatrix.getAccessCount();
	}

	/// Zero out the bias derivative parameters
	void resetDerivateOfBias()
	{
		m_db_dParams = RealZeroVector( base_type::m_kernel->numberOfParameters() + 1 );
	}

	/// Compute bias
	/// @return the bias
	double computeBiasAndSetItsDerivate(
		const RealVector& alpha,
		const RealVector& gradient,
		const RealVector& lower,
		const RealVector& upper,
		const LabeledData<InputType, unsigned int>& dataset)
	{
		// In the rare case that there are only bounded SVs and no free SVs,
		// we provide the derivative of b w.r.t. hyperparameters for external use.
		// Next, let us compute it and store it to the local variable 'm_db_dParams'

		// Call reset(again) for safety
		resetDerivateOfBias();

		// Compute the offset from the KKT conditions
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;

		// No reason to init to 0, but avoid compiler warnings
		//~ std::size_t lower_i = 0;
		//~ std::size_t upper_i = 0;
		for (std::size_t i = 0; i < gradient.size(); i++)
		{
			// In case of no free SVs, we are looking for the largest gradient of all alphas at the lower bound
			// and the smallest gradient of all alphas at the upper bound
			const double value = gradient(i);
			if (alpha(i) == lower(i))
			{

				if (value > lowerBound) {
					lowerBound = value;
					//~ lower_i = i;
				}
			}
			else if (alpha(i) == upper(i))
			{
				if (value < upperBound) {
					upperBound = value;
					//~ upper_i = i;
				}
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}

		if (freeVars > 0) {
			// Stabilized (averaged) exact value
			return sum / freeVars;
		} else {
			// TODO: need work out how to do the calculation of the derivative with missing features

			// Return best estimate
			return 0.5 * (lowerBound + upperBound);
		}
	}
};

} // namespace shark {

#endif
