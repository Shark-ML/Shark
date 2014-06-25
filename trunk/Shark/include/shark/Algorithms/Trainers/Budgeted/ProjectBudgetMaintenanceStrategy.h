//===========================================================================
/*!
 *
 *
 * \brief       Remove budget maintenance strategy
 *
 * \par
 * This is an budget strategy that simply removes one of the
 * budget vectors. Depending on the flavor, this can be e.g.
 * a random one, the smallest one (w.r.t. to alpha), etc.
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


#ifndef SHARK_MODELS_PROJECTBUDGETMAINTENANCESTRATEGY_H
#define SHARK_MODELS_PROJECTBUDGETMAINTENANCESTRATEGY_H

#include <shark/Models/Converter.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>


namespace shark
{

///
/// \brief Budget maintenance strategy that removes a vector
///
/// This is an budget strategy that simply removes one of the
/// budget vectors. Depending on the flavor, this can be e.g.
/// a random one, the smallest one (w.r.t. to alpha), etc.
///
template<class InputType>
class ProjectBudgetMaintenanceStrategy: public AbstractBudgetMaintenanceStrategy<InputType>
{
	typedef KernelExpansion<InputType> ModelType;
	typedef LabeledData<InputType, unsigned int> DataType;
	typedef typename DataType::element_type ElementType;

public:

	enum RemoveStrategyFlavor {RANDOM, SMALLEST};


	ProjectBudgetMaintenanceStrategy()
	{
	}

	/*
	 virtual void addToModel (ModelType& model, DataType newData)
	 {
	     //
	 }
	*/

	virtual void addToModel(ModelType& model, InputType const& alpha, ElementType const& supportVector)
	{
		//
	}



	std::string name() const
	{ return "ProjectBudgetMaintenanceStrategy"; }

protected:

};


}
#endif
