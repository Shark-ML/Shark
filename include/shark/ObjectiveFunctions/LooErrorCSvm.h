/*!
 *
 *  \brief Leave-one-out error for C-SVMs
 *
 *  \author T.Glasmachers
 *  \date 2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_LOOERRORCSVM_H
#define SHARK_OBJECTIVEFUNCTIONS_LOOERRORCSVM_H


#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Algorithms/QP/QpSvmDecomp.h>
#include <shark/Algorithms/QP/QpBoxDecomp.h>


namespace shark {

///
/// \brief Leave-one-out error, specifically optimized for C-SVMs.
///
template<class InputType, class CacheType = float>
class LooErrorCSvm : public SupervisedObjectiveFunction<InputType, unsigned int>
{
public:

	//////////////////////////////////////////////////////////////////
	// The types below define the type used for caching kernel values. The default is float,
	// since this type offers sufficient accuracy in the vast majority of cases, at a memory
	// cost of only four bytes. However, the type definition makes it easy to use double instead
	// (e.g., in case high accuracy training is needed).
	typedef CacheType QpFloatType;
	typedef blas::matrix<QpFloatType> QpMatrixType;
	typedef blas::matrix_row<QpMatrixType> QpMatrixRowType;
	typedef blas::matrix_column<QpMatrixType> QpMatrixColumnType;

protected:

	typedef KernelMatrix<InputType, QpFloatType> KernelMatrixType;
	typedef CachedMatrix< KernelMatrixType > CachedMatrixType;

	typedef SupervisedObjectiveFunction<InputType, unsigned int> base_type;
	typedef LabeledData<InputType, unsigned int> DatasetType;
	typedef CSvmTrainer<InputType, QpFloatType> TrainerType;
	typedef AbstractKernelFunction<InputType> KernelType;

	const DatasetType* mep_dataset;
	IParameterizable* mep_meta;
	TrainerType* mep_trainer;
	bool m_withOffset;

public:
	/// \brief Constructor.
	///
	/// \par
	/// Don't forget to call setDataset before using the object.
	LooErrorCSvm(TrainerType* trainer, bool withOffset)
	: mep_dataset(NULL)
	, mep_trainer(trainer)
	, m_withOffset(withOffset)
	{
		base_type::m_features |= base_type::HAS_VALUE;
		base_type::m_name = "LooErrorCSvm";
	}

	/// \brief Constructor.
	LooErrorCSvm(DatasetType const& dataset, TrainerType* trainer, bool withOffset)
	: mep_dataset(&dataset)
	, mep_trainer(trainer)
	, m_withOffset(withOffset)
	{
		base_type::m_features |= base_type::HAS_VALUE;
		base_type::m_name = "LooErrorCSvm";
	}


	/// inherited from SupervisedObjectiveFunction
	void setDataset(DatasetType const& dataset) {
		mep_dataset = &dataset;
	}
	
	std::size_t numberOfVariables()const{
		return mep_trainer->numberOfParameters();
	}

	/// Evaluate the leave-one-out error.
	double eval() const {
		SHARK_ASSERT(mep_dataset != NULL);
		this->m_evaluationCounter++;

		double const C = mep_trainer->C();
		KernelType* const kernel = mep_trainer->kernel();
		ZeroOneLoss<unsigned int, RealVector> loss;

		// prepare the quadratic program
		std::size_t ell = mep_dataset->numberOfElements();
		RealVector linear(ell, 0.0);
		RealVector lower(ell, 0.0);
		RealVector upper(ell, 0.0);
		RealVector alpha(ell, 0.0);
		for (std::size_t i=0; i<ell; i++)
		{
			if (mep_dataset->element(i) .label== 0)
			{
				linear(i) = -1.0;
				lower(i) = -C;
				upper(i) = 0.0;
			}
			else
			{
				SHARK_CHECK(mep_dataset->element(i).label == 1, "[LooErrorCSvm] dataset is not a binary classification problem");
				linear(i) = 1.0;
				lower(i) = 0.0;
				upper(i) = C;
			}
		}

		KernelMatrixType km(*kernel, mep_dataset->inputs());
		CachedMatrixType matrix(&km);
		QpStoppingCondition stop;

		if (m_withOffset)
		{
			// solve the problem with equality constraint
			QpSvmDecomp< CachedMatrixType > solver(matrix);
			solver.solve(linear, lower, upper, alpha, stop);

			// leave-one-out
			solver.setShrinking(false);
			double mistakes = 0;
			RealVector loo_alpha = alpha;
			for (std::size_t i=0; i<ell; i++)
			{
				// use sparseness of the solution:
				if (alpha(i) == 0.0) continue;

				// remove the i-th example
				double diff = -loo_alpha(i);
				double loo_lower = lower(i);
				double loo_upper = upper(i);
				lower(i) = 0.0;
				upper(i) = 0.0;
				if (diff > 0.0)
				{
					for (std::size_t j=0; j<ell; j++)
					{
						if (j == i) continue;
						double space = loo_alpha(j) - lower(j);
						if (space > 0.0)
						{
							if (space >= diff)
							{
								solver.modifyStep(i, j, diff);
								break;
							}
							else
							{
								diff -= space;
								solver.modifyStep(i, j, space);
							}
						}
					}
				}
				else if (diff < 0.0)
				{
					for (std::size_t j=0; j<ell; j++)
					{
						if (j == i) continue;
						double space = upper(j) - loo_alpha(j);
						if (space > 0.0)
						{
							if (space >= -diff)
							{
								solver.modifyStep(i, j, diff);
								break;
							}
							else
							{
								diff += space;
								solver.modifyStep(i, j, -space);
							}
						}
					}
				}
				solver.modifyBoxConstraints(lower, upper);

				// solve the reduced problem
				solver.warmStart(loo_alpha, stop);
				RealVector gradient = solver.getGradient();
				double b = computeB(lower, upper, loo_alpha, gradient);

				// predict
				unsigned int target = mep_dataset->element(i).label;
				RealVector prediction(1, solver.computeInnerProduct(i, loo_alpha) + b);
				double l = loss(target, prediction);
				mistakes += (l >= 1);

				// revive the i-th example
				lower(i) = loo_lower;
				upper(i) = loo_upper;
			}
			return mistakes / (double)ell;
		}
		else
		{
			// solve the problem without equality constraint
			QpBoxDecomp< CachedMatrixType > solver(matrix);
			solver.solve(linear, lower, upper, alpha, stop);

			// leave-one-out
			solver.setShrinking(false);
			double mistakes = 0;
			RealVector loo_alpha = alpha;
			for (std::size_t i=0; i<ell; i++)
			{
				// use sparseness of the solution:
				if (alpha(i) == 0.0) continue;

				// remove the i-th example
				double loo_lower = lower(i);
				double loo_upper = upper(i);
				lower(i) = 0.0;
				upper(i) = 0.0;
				solver.modifyStepTo(i, 0.0);
				solver.modifyBoxConstraints(lower, upper);

				// solve the reduced problem
				solver.warmStart(loo_alpha, stop);

				// predict
				unsigned int target = mep_dataset->element(i).label;
				RealVector prediction(1, solver.computeInnerProduct(i, loo_alpha));
				mistakes += loss(target, prediction);

				// undo the changes
				lower(i) = loo_lower;
				upper(i) = loo_upper;
			}
			return mistakes / (double)ell;
		}
	}

	/// Evaluate the leave-one-out error for the given
	/// parameters, passed to the trainer object. These
	/// parameters describe the regularization constant
	/// and the kernel parameters.
	double eval(const RealVector& parameters) const {
		SHARK_ASSERT(mep_meta != NULL);
		mep_trainer->setParameterVector(parameters);
		return eval();
	}

protected:
	/// Compute the SVM offset term (b).
	static double computeB(RealVector const& lower, RealVector const& upper, RealVector const& alpha, RealVector const& gradient)
	{
		double lowerBound = -1e100;
		double upperBound = 1e100;
		double sum = 0.0;
		std::size_t freeVars = 0;
		std::size_t ell = alpha.size();
		for (std::size_t i=0; i<ell; i++)
		{
			double value = gradient(i);
			if (alpha(i) == lower(i))
			{
				if (value > lowerBound) lowerBound = value;
			}
			else if (alpha(i) == upper(i))
			{
				if (value < upperBound) upperBound = value;
			}
			else
			{
				sum += value;
				freeVars++;
			}
		}
		if (freeVars > 0) return sum / freeVars;
		else return 0.5 * (lowerBound + upperBound);
	}
};


}
#endif
