//===========================================================================
/*!
 * 
 * \file        AbstractCost.h
 *
 * \brief       cost function for quantitative judgement of deviations of predictions from target values
 * 
 * 
 *
 * \author      T. Glasmachers
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTCOST_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTCOST_H


#include <shark/LinAlg/Base.h>
#include <shark/Core/INameable.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Core/Flags.h>
#include <shark/Data/Dataset.h>

namespace shark {


/// \brief Cost function interface
///
/// \par
/// In Shark a cost function encodes the severity of a deviation
/// of predictions from targets. This concept is more general than
/// that or a loss function, because it does not necessarily amount
/// to (uniformly) averaging a loss function over samples.
/// In general, the loss depends on the true (training) label and
/// the prediction in a not necessarily symmetric way. Also, in
/// the most general case predictions can be in a different format
/// than labels. E.g., the model prediction could be a probability
/// distribution, while the label is a single value.
///
/// \par
/// The concept of an AbstractCost function is different from that
/// encoded by the ErrorFunction class. A cost function compares
/// model predictions to labels. It does not know about the model
/// making the predictions, and thus it can not handle LabeledData
/// directly. However, it is one of the components necessary to
/// process LabeledData in an ErrorFunction.
///
template<class LabelT, class OutputT = LabelT>
class AbstractCost : public INameable, public IConfigurable
{
public:
	typedef OutputT OutputType;
	typedef LabelT LabelType;
	typedef typename Batch<OutputType>::type BatchOutputType;
	typedef typename Batch<LabelType>::type BatchLabelType;

	virtual ~AbstractCost()
	{ }

	/// list of features a cost function can have
	enum Feature {
		HAS_FIRST_DERIVATIVE = 1,
		HAS_SECOND_DERIVATIVE = 2,
		IS_LOSS_FUNCTION = 4,
	};

	SHARK_FEATURE_INTERFACE;

	/// returns true when the first parameter derivative is implemented
	bool hasFirstDerivative() const{ 
		return m_features & HAS_FIRST_DERIVATIVE; 
	}
	//~ /// returns true when the second parameter derivative is implemented
	//~ bool hasSecondDerivative() const{ 
		//~ return m_features & HAS_SECOND_DERIVATIVE; 
	//~ }
	
	/// returns true when the cost function is in fact a loss function
	bool isLossFunction() const{ 
		return m_features & IS_LOSS_FUNCTION; 
	}

	/// \brief configures the loss using informations in the property tree
	///
	/// \par
	/// The default implementation is empty and actually most cost functions
	/// do not need to be configured. See the specific documentation of the
	/// corresponding sub-classes for more details.
	/// \param  node  the configuration node
	void configure(PropertyTree const& node)
	{ (void) node; }

	/// Evaluates the cost of predictions, given targets.
	/// \param  targets      target values
	/// \param  predictions  predictions, typically made by a model
	virtual double eval(Data<LabelType> const& targets, Data<OutputType> const& predictions) const = 0;

	/// \brief evaluates the cost and the derivative w.r.t. the prediction
	///
	/// \par
	/// The default implementations throws an exception.
	/// If you overwrite this method, don't forget to set
	/// the flag HAS_FIRST_DERIVATIVE.
	/// \param  targets      target values
	/// \param  predictions  predictions, typically made by a model
	/// \param  gradient     the gradient of the cost function with respect to the predictions
	virtual double evalDerivative(
				Data<LabelType> const& targets,
				Data<OutputType> const& predictions,
				Data<OutputType>& gradient
	)const{
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_DERIVATIVE);
		return 0.0;  // dead code, prevent warning
	}
	
	double operator () (Data<LabelType> const& targets, Data<OutputType> const& predictions) const
	{ return eval(targets, predictions); }
};


}
#endif
