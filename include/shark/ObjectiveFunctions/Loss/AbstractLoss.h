//===========================================================================
/*!
 * 
 *
 * \brief       super class of all loss functions
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2010-2011
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_LOSS_ABSTRACTLOSS_H
#define SHARK_OBJECTIVEFUNCTIONS_LOSS_ABSTRACTLOSS_H

#include <shark/ObjectiveFunctions/AbstractCost.h>
#include <shark/LinAlg/Base.h>
namespace shark {


/// \brief Loss function interface
///
/// \par
/// In statistics and machine learning, a loss function encodes
/// the severity of getting a label wrong. This is am important
/// special case of a cost function (see AbstractCost), where
/// the cost is computed as the average loss over a set, also
/// known as (empirical) risk.
///
/// \par
/// It is generally agreed that loss values are non-negative,
/// and that the loss of correct prediction is zero. This rule
/// is not formally checked, but instead left to the various
/// sub-classes.
///
template<class LabelT, class OutputT = LabelT>
class AbstractLoss : public AbstractCost<LabelT, OutputT>
{
public:
	typedef AbstractCost<LabelT, OutputT> base_type;
	typedef OutputT OutputType;
	typedef LabelT LabelType;
	typedef typename VectorMatrixTraits<OutputType>::DenseMatrixType MatrixType;

	typedef typename Batch<OutputType>::type BatchOutputType;
	typedef typename Batch<LabelType>::type BatchLabelType;

	AbstractLoss(){
		this->m_features |= base_type::IS_LOSS_FUNCTION;
	}

	/// \brief evaluate the loss for a batch of targets and a prediction
	///
	/// \param  target      target values
	/// \param  prediction  predictions, typically made by a model
	virtual double eval( BatchLabelType const& target, BatchOutputType const& prediction) const = 0;

	/// \brief evaluate the loss for a target and a prediction
	///
	/// \param  target      target value
	/// \param  prediction  prediction, typically made by a model
	virtual double eval( LabelType const& target, OutputType const& prediction)const{
		BatchLabelType labelBatch = Batch<LabelType>::createBatch(target,1);
		get(labelBatch,0)=target;
		BatchOutputType predictionBatch = Batch<OutputType>::createBatch(prediction,1);
		get(predictionBatch,0)=prediction;
		return eval(labelBatch,predictionBatch);
	}

	/// \brief evaluate the loss and its derivative for a target and a prediction
	///
	/// \param  target      target value
	/// \param  prediction  prediction, typically made by a model
	/// \param  gradient    the gradient of the loss function with respect to the prediction
	virtual double evalDerivative(LabelType const& target, OutputType const& prediction, OutputType& gradient) const {
		BatchLabelType labelBatch = Batch<LabelType>::createBatch(target,1);
		get(labelBatch, 0) = target;
		BatchOutputType predictionBatch = Batch<OutputType>::createBatch(prediction, 1);
		get(predictionBatch, 0) = prediction;
		BatchOutputType gradientBatch = Batch<OutputType>::createBatch(gradient, 1);
		double ret = evalDerivative(labelBatch, predictionBatch, gradientBatch);
		gradient = get(gradientBatch, 0);
		return ret;
	}
	
	/// \brief evaluate the loss and its first and second derivative for a target and a prediction
	///
	/// \param  target      target value
	/// \param  prediction  prediction, typically made by a model
	/// \param  gradient    the gradient of the loss function with respect to the prediction
	/// \param  hessian     the hessian of the loss function with respect to the prediction
	virtual double evalDerivative(
		LabelType const& target, OutputType const& prediction,
		OutputType& gradient,MatrixType & hessian
	) const {
		SHARK_FEATURE_EXCEPTION_DERIVED(HAS_SECOND_DERIVATIVE);
		return 0.0;  // dead code, prevent warning
	}

	/// \brief evaluate the loss and the derivative w.r.t. the prediction
	///
	/// \par
	/// The default implementations throws an exception.
	/// If you overwrite this method, don't forget to set
	/// the flag HAS_FIRST_DERIVATIVE.
	/// \param  target      target value
	/// \param  prediction  prediction, typically made by a model
	/// \param  gradient    the gradient of the loss function with respect to the prediction
	virtual double evalDerivative(BatchLabelType const& target, BatchOutputType const& prediction, BatchOutputType& gradient) const
	{
		SHARK_FEATURE_EXCEPTION_DERIVED(HAS_FIRST_DERIVATIVE);
		return 0.0;  // dead code, prevent warning
	}

	//~ /// \brief evaluate the loss and fist and second derivative w.r.t. the prediction
	//~ ///
	//~ /// \par
	//~ /// The default implementations throws an exception.
	//~ /// If you overwrite this method, don't forget to set
	//~ /// the flag HAS_FIRST_DERIVATIVE.
	//~ /// \param  target      target value
	//~ /// \param  prediction  prediction, typically made by a model
	//~ /// \param  gradient    the gradient of the loss function with respect to the prediction
	//~ /// \param  hessian      the hessian matrix of the loss function with respect to the prediction
	//~ virtual double evalDerivative(
			//~ LabelType const& target,
			//~ OutputType const& prediction,
			//~ OutputType& gradient,
			//~ MatrixType& hessian) const
	//~ {
		//~ SHARK_FEATURE_EXCEPTION_DERIVED(HAS_SECOND_DERIVATIVE);
		//~ return 0.0;  // dead code, prevent warning
	//~ }

	/// from AbstractCost
	///
	/// \param  targets      target values
	/// \param  predictions  predictions, typically made by a model
	double eval(Data<LabelType> const& targets, Data<OutputType> const& predictions) const{
		SIZE_CHECK(predictions.numberOfElements() == targets.numberOfElements());
		SIZE_CHECK(predictions.numberOfBatches() == targets.numberOfBatches());
		int numBatches = (int) targets.numberOfBatches();
		double error = 0;
		SHARK_PARALLEL_FOR(int i = 0; i < numBatches; ++i){
			double batchError= eval(targets.batch(i),predictions.batch(i));
			SHARK_CRITICAL_REGION{
				error+=batchError;
			}
		}
		return error / targets.numberOfElements();
	}

	/// from AbstractCost
	///
	/// \param  targets      target values
	/// \param  predictions  predictions, typically made by a model
	/// \param  gradient     the gradient of the cost function with respect to the predictions
	double evalDerivative(
		Data<LabelType> const& targets,
		Data<OutputType> const& predictions,
		Data<OutputType>& gradient
	) const{
		SHARK_FEATURE_EXCEPTION_DERIVED(HAS_FIRST_DERIVATIVE);
		SIZE_CHECK(predictions.numberOfElements() == targets.numberOfElements());

		int numBatches = (int) targets.numberOfBatches();
		std::size_t elements = targets.numberOfElements();
		gradient = Data<OutputType>(numBatches);
		
		double error = 0;
		SHARK_PARALLEL_FOR(int i = 0; i < numBatches; ++i){
			double batchError= evalDerivative(targets.batch(i),predictions.batch(i),gradient.batch(i));
			gradient.batch(i) /= elements;//we return the mean of the loss
			SHARK_CRITICAL_REGION{
				error+=batchError;
			}
		}
		return error/elements;
	}

	/// \brief evaluate the loss for a target and a prediction
	///
	/// \par
	/// convenience operator
	///
	/// \param  target      target value
	/// \param  prediction  prediction, typically made by a model
	double operator () (LabelType const& target, OutputType const& prediction) const
	{ return eval(target, prediction); }

	double operator () (BatchLabelType const& target, BatchOutputType const& prediction) const
	{ return eval(target, prediction); }

	using base_type::operator();
};


}
#endif
