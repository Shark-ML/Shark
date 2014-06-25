//===========================================================================
/*!
 *
 *
 * \brief       Merge budget maintenance strategy
 *
 * \par
 * This is an budget strategy that adds a new vector by merging
 * a pair of budget vectors. The pair to merge is found by first
 * searching for the budget vector with the smallest alpha-coefficients
 * (measured in 2-norm), and then finding the second one by
 * computing a certain degradation measure. This is therefore linear
 * in the size of the budget.
 *
 * \par
 * The method is an implementation of the merge strategy
 * given in wang, crammer, vucetic: "Breaking the Curse of Kernelization:
 * Budgeted Stochastic Gradient Descent for Large-Scale SVM Training"
 * and owes very much to the implementation in BudgetedSVM.
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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


#ifndef SHARK_MODELS_MERGEBUDGETMAINTENANCESTRATEGY_H
#define SHARK_MODELS_MERGEBUDGETMAINTENANCESTRATEGY_H

#include <shark/Algorithms/GradientDescent/LineSearch.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>

#include <shark/Algorithms/Trainers/Budgeted/AbstractBudgetMaintenanceStrategy.h>


namespace shark
{

///
/// \brief Budget maintenance strategy that merges two vectors
///
/// \par This is an budget strategy that simply merges two budget vectors
/// in order to make space for a new one. This is done by first searching
/// for the budget vector that has smallest \f[ ||\alpha||_2\f] coefficient-- only
/// then a second one is searched for, by inspecting the expected
/// weight degradation after merging. The vector with smallest weight
/// degradation is the vector one should merge with the first one.
/// By this heuristic, the merging strategy has complexity \f[ \mathcal{O}(B) \f].
/// Compared with the projection strategy, merging should be faster, and stil
/// obtains similar accuracy. Unluckily any kind of timing numbers are missing
/// in the reference paper of Wang, Crammer and Vucetic.
///
/// \par Note that in general it is unclear how two data objects should be merged,
/// e.g. strings must be merged differently than vectors. Therefore it is necessary
/// to create an specialization of this strategy for a given input type.
///
template<class InputType>
class MergeBudgetMaintenanceStrategy: public AbstractBudgetMaintenanceStrategy<InputType>
{
	typedef KernelExpansion<InputType> ModelType;
	typedef LabeledData<InputType, unsigned int> DataType;
	typedef typename DataType::element_type ElementType;

public:

	/// constructor.
	MergeBudgetMaintenanceStrategy()
	{
	}


	/// add to model.
	/// this is just a fake here, as it is unclear in general how to merge two objects,
	/// one needs to specialize this template.
	///
	/// @param[in,out]  model   the model the strategy will work with
	/// @param[in]  alpha   alphas for the new budget vector
	/// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
	///
	virtual void addToModel(ModelType& model, InputType &alpha, ElementType &supportVector)
	{
		// this has to be implemented for every specialization
		// so here we throw some error
		throw(SHARKEXCEPTION("MergeBudgetMaintenanceStrategy: There is no default merging strategy for the InputType you provided! Please specialize this class."));
	}


	/// class name
	std::string name() const
	{ return "MergeBudgetMaintenanceStrategy"; }
};



///
/// \brief Budget maintenance strategy merging vectors.
///
/// \par This is an specialization of the merge budget maintenance strategy
/// that handles simple real-valued vectors. This is a nearly 1:1 adoption of
/// the strategy presented in Wang, Cramer and Vucetic.
///
template<>
class MergeBudgetMaintenanceStrategy<RealVector>: public AbstractBudgetMaintenanceStrategy<RealVector>
{
	typedef KernelExpansion<RealVector> ModelType;
	typedef LabeledData<RealVector, unsigned int> DataType;
	typedef typename DataType::element_type ElementType;
	typedef RealVector InputType;

public:

	/// constructor.
	MergeBudgetMaintenanceStrategy()
	{
	}


	/// This is the objective function we need to optimize during merging.
	/// Basically the merging strategy needs a line search to find the parameter
	/// which maximizes \f[ a \cdot k_h (x_m, x_n) + b \cdot k_{1-h}(x_m, x_n) \f].
	/// (all in the notation of wang, crammer and vucetic).
	/// The coefficients a and b  are given by the alpha coefficients of the
	/// corresponding support vectors  \f[ x_m \f] and  \f[ x_n\f], more precicely
	/// we have \f[ a = \sum \alpha_m^{(i)}/d_i\f] and \f[b = 1 - a = \sum \alpha_n^{(i)}/d_i\f]
	/// with \f[d_i = \alpha_m^{(i)} + \alpha_n^{(i)} \f].
	///
	struct MergingProblemFunction : public SingleObjectiveFunction
	{
		typedef SingleObjectiveFunction Base;

		/// class name
		std::string name() const
		{ return "MergingProblemFunction"; }


		/// parameters for the function.
		double m_a, m_b;
		double m_k; //< contains (in our problem) k(x_m, x_n)


		/// constructor.
		/// \param[in] a    a coefficient of the formula
		/// \param[in] b    b coefficient of the formula
		/// \param[in] k    k coefficient of the formula
		///
		MergingProblemFunction(double a, double b, double k)
		{
			m_a = a;
			m_b = b;
			m_k = k;
		}


		/// number of variables, we have a one-dimensional problem here.
		std::size_t numberOfVariables()const
		{
			return 1;
		}


		/// evaluation
		/// \param[in]  pattern      vector to evaluate the function at. as we have a 1d problem,
		///                                         we ignore everything beyond the first component.
		/// \return function value at the point
		///
		virtual double eval(RealVector const& pattern)const
		{
			double h = pattern(0);
			// we want to maximize, thus minimize   -function
			return (- (m_a * pow(m_k, (1.0 - h) * (1.0 - h)) + m_b * pow(m_k, h * h)));
		}


		/// Derivative of function.
		/// Unsure if the derivative is really needed, but wolfram alpha
		/// helped computing it, do not want to let it down, wasting its capacity.
		/// The search routine uses it, as we did not removed the derivative-feature.
		/// \param[in]  input   Point to evaluate the function at
		/// \param[out] derivative  Derivative at the given point
		/// \return     Function value at the point
		///
		virtual double evalDerivative(const SearchPointType & input, FirstOrderDerivative & derivative)const
		{
			double h = input(0);
			// we want to maximize, thus minimize   -function
			derivative(0) = 2 * log(m_k) * (-m_a * (h - 1.0) * pow(m_k, (h - 1.0) * (h - 1.0))
											- m_b * h * pow(m_k, (h * h)));
			return eval(input);
		}
	};



	/// Reduce the budget.
	/// This is a helper routine. after the addToModel adds the new support vector
	/// to the end of the budget (it was chosen one bigger than the capacity),
	/// this routine will do the real merging. Given a index it will search for a second
	/// index, so that merging is 'optimal'. It then will perform the merging. After that
	/// the last budget vector will be freed again (by setting its alpha-coefficients to zero).
	/// \param[in]  model   Model to work on
	/// \param[in]  firstIndex  The index of the first element of the pair to merge.
	///
	virtual void reduceBudget(ModelType& model, size_t firstIndex)
	{
		size_t maxIndex = model.basis().numberOfElements();

		// compute the kernel row of the given, first element and all the others
		// should take O(B) time, as it is a row of size B
		blas::vector<float> kernelRow(maxIndex, 0.0);
		for(size_t j = 0; j < maxIndex; j++)
			kernelRow(j) = model.kernel()->eval(model.basis().element(firstIndex), model.basis().element(j));

		// initialize the search
		double fret(0.);
		RealVector h(1);     // initial search starting point
		RealVector xi(1);    // direction of search

		// save the parameter at the minimum
		double minDegradation = std::numeric_limits<double>::infinity();
		double minH = 0.0;
		double minAlphaMergedFirst = 0.0;
		double minAlphaMergedSecond = 0.0;
		size_t secondIndex = 0;


		// we need to check every other vector
		RealMatrix &alpha = model.alpha();
		for(size_t currentIndex = 0; currentIndex < maxIndex; currentIndex++)
		{
			// we do not want the vector already chosen
			if(firstIndex == currentIndex)
				continue;

			// compute the alphas for the model, this is the formula
			// between (6.7) and (6.8) in wang, crammer, vucetic
			double a = 0.0;
			double b = 0.0;
			for(size_t c = 0; c < alpha.size2(); c++)
			{
				double d = std::min(0.00001, alpha(currentIndex, c) + alpha(firstIndex, c));
				a += alpha(firstIndex, c) / d;
				b += alpha(currentIndex, c) / d;
			}

			// Initialize search starting point and direction:
			h(0) = 0.0;
			xi(0) = 0.5;

			double k = kernelRow(currentIndex);
			MergingProblemFunction mergingProblemFunction(a, b, k);

			// minimize function
			// search between 0 and 1
			detail::dlinmin(h, xi, fret, mergingProblemFunction, 0.0, 1.0);

			// the optimal point is now given by h.
			// the vector that corresponds to this is
			// $z = h x_m + (1-h) x_n$  by formula (6.7)
			RealVector firstVector = model.basis().element(firstIndex);
			RealVector currentVector = model.basis().element(currentIndex);
			RealVector mergedVector = h(0) * firstVector + (1.0 - h(0)) * currentVector;

			// this is another minimization problem, which has as optimal
			// solution $\alpha_z^{(i)} = \alpha_m^{(i)} k(x_m, z) + \alpha_n^{(i)} k(x_n, z).$

			// the coefficient of this z is computed by approximating
			// both vectors by the merged one. maybe KernelBasisDistance can be used
			// but i am not sure, if at all and if, if its faster.

			long double alphaMergedFirst = pow(k, (1.0 - h(0)) * (1.0 - h(0)));
			long double alphaMergedCurrent = pow(k, h(0) * h(0));

			// degradation is computed for each class
			// this is computed by using formula (6.8), applying it to each class and summing up
			// here a kernel with $k(x,x) = 1$ is assumed
			double currentDegradation = 0.0f;
			for(size_t c = 0; c < alpha.size2(); c++)
			{
				double zAlpha = alphaMergedFirst * alpha(firstIndex, c) + alphaMergedCurrent * alpha(currentIndex, c);
				// TODO: unclear to me why this is the thing we want to compute
				currentDegradation += pow(alpha(firstIndex, c), 2) + pow(alpha(currentIndex, c), 2) +
									  2.0 * k * alpha(firstIndex, c) * alpha(currentIndex, c) - zAlpha * zAlpha;
			}

			// TODO: this is shamelessly copied, as the rest, but maybe i want to refactor it and make it nicer.
			if(currentDegradation < minDegradation)
			{
				minDegradation = currentDegradation;
				minH = h(0);
				minAlphaMergedFirst = alphaMergedFirst;
				minAlphaMergedSecond = alphaMergedCurrent;
				secondIndex = currentIndex;
			}
		}

		// compute merged vector
		RealVector firstVector = model.basis().element(firstIndex);
		RealVector secondVector = model.basis().element(secondIndex);
		RealVector mergedVector = minH * firstVector + (1.0 - minH) * secondVector;

		// replace the second vector by the merged one
		model.basis().element(secondIndex) = mergedVector;

		// and update the alphas
		for(size_t c = 0; c < alpha.size2(); c++)
		{
			alpha(secondIndex, c) = minAlphaMergedFirst * alpha(firstIndex, c) + minAlphaMergedSecond * alpha(secondIndex, c);
		}

		// the first index is now obsolete, so we copy the
		// last vector, which serves as a buffer, to this position
		row(alpha, firstIndex) = row(alpha, maxIndex - 1);
		model.basis().element(firstIndex) = model.basis().element(maxIndex - 1);

		// clear the  buffer by cleaning the alphas
		// finally the vectors we merged.
		row(model.alpha(), maxIndex - 1).clear();
	}



	/// add a vector to the model.
	/// this will add the given vector to the model and merge the budget so that afterwards
	/// the budget size is kept the same. If the budget has a free entry anyway, no merging
	/// will be performed, but instead the given vector is simply added to the budget.
	///
	/// @param[in,out]  model   the model the strategy will work with
	/// @param[in]  alpha   alphas for the new budget vector
	/// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
	///
	virtual void addToModel(ModelType& model, InputType const& alpha, ElementType const& supportVector)
	{

		// find the two indicies we want to merge

		// note that we have to crick ourselves, as the model has
		// a fixed size, but actually we want to work with the model
		// together with the new supportvector. so our budget size
		// is one greater than the user specified and we use this
		// last entry of the model for buffer. it will be freed again,
		// when merging is finished.

		// put the new vector into place
		size_t maxIndex = model.basis().numberOfElements();
		model.basis().element(maxIndex - 1) = supportVector.input;
		row(model.alpha(), maxIndex - 1) = alpha;


		// the first vector to merge is the one with the smallest alpha coefficient
		// (it cannot be our new vector, because in each iteration the
		// old weights get downscaled and the new ones get the biggest)
		size_t firstIndex = 0;
		double firstAlpha = 0;
		findSmallestVector(model, firstIndex, firstAlpha);

		// if the smallest vector has zero alpha,
		// the budget is not yet filled so we can skip merging it.
		if(firstAlpha == 0.0f)
		{
			// as we need the last vector to be zero, we put the new
			// vector to that place and undo our putting-the-vector-to-back-position
			model.basis().element(firstIndex) = supportVector.input;
			row(model.alpha(), firstIndex) = alpha;

			// enough to zero out the alpha
			row(model.alpha(), maxIndex - 1).clear();

			// ready.
			return;
		}

		// the second one is given by searching for the best match now,
		// taking O(B) time. we also have to provide to the findVectorToMerge
		// function the supportVector we want to add, as we cannot, as
		// said, just extend the model with this vector.
		reduceBudget(model, firstIndex);
	}


	/// class name
	std::string name() const
	{ return "MergeBudgetMaintenanceStrategy"; }


protected:
};

}
#endif
