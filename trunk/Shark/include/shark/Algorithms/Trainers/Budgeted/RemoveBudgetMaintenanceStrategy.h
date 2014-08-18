//===========================================================================
/*!
 *
 *
 * \brief       Remove budget maintenance strategy.
 *
 * \par
 * This is an budget strategy that simply removes one of the
 * budget vectors. Depending on the flavor, this can be e.g.
 * a random one, the smallest one (w.r.t. to 2-norm of the alphas)
 *
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


#ifndef SHARK_MODELS_REMOVEBUDGETMAINTENANCESTRATEGY_H
#define SHARK_MODELS_REMOVEBUDGETMAINTENANCESTRATEGY_H

#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>

#include <shark/Algorithms/Trainers/Budgeted/AbstractBudgetMaintenanceStrategy.h>


namespace shark
{

///
/// \brief Budget maintenance strategy that removes a vector
///
/// This is an budget strategy that simply removes one of the
/// budget vectors. Depending on the flavor, this can be e.g.
/// a random one, the smallest one (w.r.t. to 2-norm of the alphas)
///
template<class InputType>
class RemoveBudgetMaintenanceStrategy: public AbstractBudgetMaintenanceStrategy<InputType>
{
	typedef KernelExpansion<InputType> ModelType;
	typedef LabeledData<InputType, unsigned int> DataType;
	typedef typename DataType::element_type ElementType;

public:

	/// the flavors of the remove strategy
	enum RemoveStrategyFlavor {RANDOM, SMALLEST};


	/// constructor.
	/// @param[in] flavor   enum that decides on the method a vector is removed.
	RemoveBudgetMaintenanceStrategy(size_t flavor = SMALLEST)
		: m_flavor(flavor)
	{
	}


	/// add a vector to the model.
	/// this will add the given vector to the model and remove another one depending on the flavor.
	///
	/// @param[in,out]  model   the model the strategy will work with
	/// @param[in]  alpha   alphas for the new budget vector
	/// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
	///
	virtual void addToModel(ModelType& model, InputType const& alpha, ElementType const& supportVector)
	{

		// first we check: if the budget is not full, we do not need to do remove anything
		std::size_t index = 0;
		double minAlpha = 0;
		this->findSmallestVector(model, index, minAlpha);

		if(minAlpha == 0.0f)
		{
			// replace vector and alpha
			model.basis().element(index) = supportVector.input;
			row(model.alpha(), index) = alpha;
			return;
		}

		// else depending on the flavor we do something
		switch(m_flavor)
		{
		case RANDOM:
		{
			// though we have found the smallest one,  we want to remove
			// a random element.
			index = Rng::discrete(0, model.basis().numberOfElements() - 1);
			break;
		}
		case SMALLEST:
		{
			// we already have found the smallest alpha, so nothing to do
			break;
		}
		default:
			// throw some error
			throw(SHARKEXCEPTION("RemoveBudgetMaintenanceStrategy: Unknown flavor!"));
		}

		// replace vector and alpha
		model.basis().element(index) = supportVector.input;
		row(model.alpha(), index) = alpha;

                // we need to clear out the last vector, as it is just a buffer
                row (model.alpha(), model.basis().numberOfElements() -1).clear();
	}


	/// class name
	std::string name() const
	{ return "RemoveBudgetMaintenanceStrategy"; }

protected:
	/// flavor for removing a vector
	size_t m_flavor;
};

}
#endif
