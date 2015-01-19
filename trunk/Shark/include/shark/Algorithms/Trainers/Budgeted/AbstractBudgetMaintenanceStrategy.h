//===========================================================================
/*!
 *
 *
 * \brief       Abstract Budget maintenance strategy
 *
 * \par
 * This holds the interface for any budget maintenance strategy.
 *
 *
 *
 *
 * \author      Aydin Demircioglu
 * \date        2014
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


#ifndef SHARK_MODELS_ABSTRACTBUDGETMAINTENANCESTRATEGY_H
#define SHARK_MODELS_ABSTRACTBUDGETMAINTENANCESTRATEGY_H

#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/KernelExpansion.h>


namespace shark
{

///
/// \brief This is the abstract interface for any budget maintenance strategy.
///
/// To allow for easy exchange of budget maintenance strategies, each of
/// them should derive from this class. The only function it defines is addToModel,
/// which, when implemented, will add a given supportvector and given alphas
/// to the provided model by applying the respective budget maintenance strategy.
/// (Note that not all merging strategies need the alphas, but some do)
///
template<class InputType>
class AbstractBudgetMaintenanceStrategy
{

public:
	typedef KernelExpansion<InputType> ModelType;
	typedef LabeledData<InputType, unsigned int> DataType;
	typedef typename DataType::element_type ElementType;

	AbstractBudgetMaintenanceStrategy()
	{ }


	/// this is the main interface, which adds a given supportvector with
	/// given alpha coefficients to the model.
	///
	/// @param[in,out]  model   the model the strategy will work with
	/// @param[in]  alpha   alphas for the new budget vector
	/// @param[in]  supportVector the vector to add to the model by applying the maintenance strategy
	///
	virtual void addToModel(ModelType& model, InputType const& alpha, ElementType const& supportVector)  = 0;



	/// this will find the vector with the smallest alpha, measured in 2-norm
	/// in the given model. now there is a special case: if there is somewhere a zero
	/// coefficient, then obviously this is the smallest element. in this case we
	/// just proceed as usual. the caller must decide what to do with such a vector.
	/// \par note: if the model is completely empty, we will give back infinity and index 0.
	/// this is again for the caller to handle. for safety, we put an assert in there.
	///
	/// @param[in]  model       the model we want to search
	/// @param[out] minIndex    the index of the vector with smallest coefficient
	/// @param[out] minAlpha    the 2-norm of the alpha coefficient of the found vector
	///
	static void findSmallestVector(ModelType const& model, size_t &minIndex, double &minAlpha)
	{
		// we do not have it, so we remove the vector with the
		// smallest 'influcence', measured by the smallest alpha

		minAlpha = std::numeric_limits<double>::infinity();
		minIndex = 0;

		for(size_t j = 0; j < model.alpha().size1(); j++)
		{
			double currentNorm = blas::norm_2(row(model.alpha(), j));

			if(currentNorm < minAlpha)
			{
				minAlpha = blas::norm_2(row(model.alpha(), j));
				minIndex = j;
			}
		}

		SHARK_ASSERT(minAlpha != std::numeric_limits<double>::infinity());
	}


	/// return the class name
	std::string name() const
	{ return "AbstractBudgetMaintenanceStrategy"; }
};


}
#endif
