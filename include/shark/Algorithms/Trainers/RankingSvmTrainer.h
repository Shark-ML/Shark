//===========================================================================
/*!
 * 
 *
 * \brief       Support Vector Machine Trainer for the ranking-SVM
 * 
 *
 * \author      T. Glasmachers
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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


#ifndef SHARK_ALGORITHMS_RANKINGSVMTRAINER_H
#define SHARK_ALGORITHMS_RANKINGSVMTRAINER_H


#include <shark/Algorithms/Trainers/AbstractSvmTrainer.h>
#include <shark/Algorithms/QP/BoxConstrainedProblems.h>
#include <shark/Algorithms/QP/SvmProblems.h>
#include <shark/Algorithms/QP/QpSolver.h>
#include <shark/LinAlg/DifferenceKernelMatrix.h>
#include <shark/LinAlg/CachedMatrix.h>
#include <shark/LinAlg/PrecomputedMatrix.h>

namespace shark {


///
/// \brief Training of an SVM for ranking.
///
/// A ranking SVM trains a function (linear or linear in a kernel
/// induced feature space, RKHS) with the aim that the function values
/// are consistent with given pairwise rankings. I.e., given are pairs
/// (a, b) of points, and the task of SVM training is to find a
/// function f such that f(a) < f(b). More exactly, the hard margin
/// ranking SVM aims for f(b) - f(a) >= 1 while minimizing the squared
/// RKHS norm of f. The soft-margin SVM relates the constraint analogous
/// to a standard C-SVM.
///
/// The trained model is a real-valued function. To predict the ranking
/// of a pair of points the function is applied to both points. The one
/// with smaller function value is ranked as being "smaller", i.e., if f
/// is the trained model and a and b are data points, then the following
/// code computes the ranking:
///
///   bool a_better_than_b = (f(a) < f(b));
///
/// \ingroup supervised_trainer
template <class InputType, class CacheType = float>
class RankingSvmTrainer : public AbstractSvmTrainer< InputType, unsigned int, KernelExpansion<InputType> >
{
private:
	typedef AbstractSvmTrainer< InputType, unsigned int, KernelExpansion<InputType> > base_type;

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

	typedef AbstractKernelFunction<InputType> KernelType;

	//! Constructor
	//! \param  kernel         kernel function to use for training and prediction
	//! \param  C              regularization parameter - always the 'true' value of C, even when unconstrained is set
	//! \param  unconstrained  when a C-value is given via setParameter, should it be piped through the exp-function before using it in the solver?
	RankingSvmTrainer(KernelType* kernel, double C, bool unconstrained = false)
	: base_type(kernel, C, false, unconstrained)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RankingSvmTrainer"; }

	/// \brief Train the ranking SVM.
	///
	/// This variant of the train function assumes that all pairs of
	/// points should be ranked according to the order they appear in
	/// the data set. 
	void train(KernelExpansion<InputType>& function, Data<InputType> const& dataset)
	{
		// create all pairs
		std::size_t n = dataset.numberOfElements();
		std::vector<std::pair<std::size_t, std::size_t>> pairs;
		for (std::size_t i=0; i<n; i++) {
			for (std::size_t j=0; j<i; j++) {
				pairs.push_back(std::make_pair(j, i));
			}
		}
		train(function, dataset, pairs);
	}

	/// \brief Train the ranking SVM.
	///
	/// This variant of the train function uses integer labels to define
	/// pairwise rankings. It is trained on all pairs of data points
	/// with different label, aiming for a smaller function value for
	/// the point with smaller label.
	void train(KernelExpansion<InputType>& function, LabeledData<InputType, unsigned int> const& dataset)
	{
		std::vector<std::pair<std::size_t, std::size_t>> pairs;
		std::size_t i = 0;
		for (auto const& yi : dataset.labels().elements()) {
			std::size_t j = 0;
			for (auto const& yj : dataset.labels().elements()) {
				if (j >= i) break;
				if (yi < yj) pairs.push_back(std::make_pair(i, j));
				else if (yi > yj) pairs.push_back(std::make_pair(j, i));
				j++;
			}
			i++;
		}
		train(function, dataset.inputs(), pairs);
	}

	/// \brief Train the ranking SVM.
	///
	/// This variant of the train function works with explicitly given
	/// pairs of data points. Each pair is identified by the indices of
	/// the training points in the data set.
	void train(KernelExpansion<InputType>& function, Data<InputType> const& dataset, std::vector<std::pair<std::size_t, std::size_t>> const& pairs)
	{
		function.setStructure(base_type::m_kernel, dataset, false);
		DifferenceKernelMatrix<InputType, QpFloatType> dm(*function.kernel(), dataset, pairs);

		if (QpConfig::precomputeKernel())
		{
			PrecomputedMatrix< DifferenceKernelMatrix<InputType, QpFloatType> > matrix(&dm);
			trainInternal(function, dataset, pairs, matrix);
		}
		else
		{
			CachedMatrix< DifferenceKernelMatrix<InputType, QpFloatType> > matrix(&dm);
			trainInternal(function, dataset, pairs, matrix);
		}
	}

private:
	template <typename MatrixType>
	void trainInternal(KernelExpansion<InputType>& function, Data<InputType> const& dataset, std::vector<std::pair<std::size_t, std::size_t>> const& pairs, MatrixType& matrix)
	{
		GeneralQuadraticProblem<MatrixType> qp(matrix);
		qp.linear = RealVector(qp.dimensions(), 1.0);
		qp.boxMin = RealVector(qp.dimensions(), 0.0);
		qp.boxMax = RealVector(qp.dimensions(), this->C());
		typedef BoxConstrainedShrinkingProblem< GeneralQuadraticProblem<MatrixType> > ProblemType;
		ProblemType problem(qp, base_type::m_shrinking);

		QpSolver<ProblemType> solver(problem);
		solver.solve(base_type::stoppingCondition(), &base_type::solutionProperties());
		RealVector alpha = problem.getUnpermutedAlpha();
		RealVector coeff(dataset.numberOfElements(), 0.0);
		SIZE_CHECK(pairs.size() == alpha.size());
		for (std::size_t i=0; i<alpha.size(); i++)
		{
			double a = alpha(i);
			coeff(pairs[i].first) -= a;
			coeff(pairs[i].second) += a;
		}
		blas::column(function.alpha(),0) = coeff;

		if (base_type::sparsify()) function.sparsify();
	}
};


}
#endif
