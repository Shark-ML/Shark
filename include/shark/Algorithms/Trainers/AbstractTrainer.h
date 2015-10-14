//===========================================================================
/*!
 * 
 *
 * \brief       Abstract Trainer Interface.
 * 
 * 
 *
 * \author      O. Krause, T.Glasmachers
 * \date        2010-2011
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
#ifndef SHARK_ALGORITHMS_TRAINERS_ABSTRACTTRAINER_H
#define SHARK_ALGORITHMS_TRAINERS_ABSTRACTTRAINER_H

#include <shark/Core/INameable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/AbstractModel.h>

namespace shark {


///
/// \brief Superclass of supervised learning algorithms
///
/// \par
/// AbstractTrainer is the super class of all trainers,
/// i.e., procedures for training or learning model
/// parameters. It provides a single virtual function to
/// train the model.
///
/// \par
/// Note: Most learning algorithms of this type operate on
/// a special model type, such as a linear model, a kernel
/// expansion, etc. Thus, these algorithms should provide
/// a specialized train method accepting only this model
/// type. The virtual train method should be overriden
/// with a method that checks the type of the model and
/// calls the specialized train method.
///
template <class Model, class LabelTypeT = typename Model::OutputType>
class AbstractTrainer: public INameable, public ISerializable
{
public:
	typedef Model ModelType;
	typedef typename ModelType::InputType InputType;
	typedef LabelTypeT LabelType;
	typedef LabeledData<InputType, LabelType> DatasetType;
	/// Core of the Trainer interface
	virtual void train(ModelType& model, DatasetType const& dataset) = 0;
};


///
/// \brief Superclass of unsupervised learning algorithms
///
/// \par
/// AbstractUnsupervisedTrainer is the superclass of all
/// unsupervised learning algorithms. It consists of a
/// single virtual function to train the model.
///
/// \par
/// Note: Most learning algorithms of this type operate on
/// a special model type, such as a linear model, a kernel
/// expansion, or a nearest neighbor model. Thus, these
/// algorithms should provide a specialized train method
/// that accepts only this model type. The virtual train
/// method should be overriden with a method that checks
/// the type of the model and calls the specialized train
/// method.
///
template <class Model>
class AbstractUnsupervisedTrainer : public INameable, public ISerializable
{
public:
	typedef Model ModelType;
	typedef typename Model::InputType InputType;
	/// Core of the Trainer interface
	virtual void train(ModelType& model, const UnlabeledData<InputType>& inputset) = 0;
};


}
#endif
