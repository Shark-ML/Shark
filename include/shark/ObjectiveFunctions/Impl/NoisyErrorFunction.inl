/*!
 *  \brief implementation of NoisyErrorFunction
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_NOISYERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_NOISYERRORFUNCTION_H

#include <shark/Core/Random.h>

namespace shark{

namespace detail{
/// \brief Implementation for the NoisyErrorFunction. It hides the Type of the OutputType.
template<class InputType,class LabelType, class OutputType>
class NoisyErrorFunctionWrapper : public FunctionWrapperBase
{
private:
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractLoss<LabelType,OutputType>* mep_loss;
	LabeledData<InputType,LabelType> m_dataset;
	typedef typename AbstractModel<InputType, OutputType>::BatchOutputType BatchOutputType;
	typedef typename LabeledData<InputType,LabelType>::batch_type BatchDataType;

public:
	NoisyErrorFunctionWrapper(
		LabeledData<InputType,LabelType> const& dataset,
		AbstractModel<InputType,OutputType>* model,
		AbstractLoss<LabelType,OutputType>* loss
	): mep_model(model), mep_loss(loss), m_dataset(dataset)
	{
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);
		
		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			this->m_features|=HAS_FIRST_DERIVATIVE;
		this->m_features|=CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NoisyErrorFunctionWrapper"; }

	FunctionWrapperBase* clone()const{
		return new NoisyErrorFunctionWrapper<InputType,LabelType,OutputType>(*this);
	}

	SearchPointType proposeStartingPoint()const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	double eval(RealVector const& input)const {
		mep_model->setParameterVector(input);
		
		std::size_t batchIndex = random::discrete(*mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
		auto const& batch = m_dataset.batch(batchIndex);
		
	
		BatchOutputType predictions;
		mep_model->eval(batch.input,predictions);

		//calculate error derivative of the loss function
		double error= mep_loss->eval(batch.label, predictions);
		error /= shark::batchSize(batch);
		return error;
	}

	ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative)const {
		mep_model->setParameterVector(input);
		
		std::size_t batchIndex = random::discrete(*mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
		auto const& batch = m_dataset.batch(batchIndex);
		
		boost::shared_ptr<State> state = mep_model->createState();
		BatchOutputType predictions;
		mep_model->eval(batch.input,predictions,*state);

		//calculate error derivative of the loss function
		BatchOutputType errorDerivative;
		double error= mep_loss->evalDerivative(batch.label, predictions,errorDerivative);

		//chain rule
		mep_model->weightedParameterDerivative(batch.input,predictions, errorDerivative,*state,derivative);
	
		error /= shark::batchSize(batch);
		derivative /= shark::batchSize(batch);
		return error;
	}
};
}

template<class InputType, class LabelType, class OutputType>
NoisyErrorFunction::NoisyErrorFunction(
	LabeledData<InputType,LabelType> const& dataset,
	AbstractModel<InputType,OutputType>* model,
	AbstractLoss<LabelType, OutputType>* loss)
:mp_wrapper(new detail::NoisyErrorFunctionWrapper<InputType,LabelType,OutputType>(dataset,model,loss))
, m_regularizer(0), m_regularizationStrength(0){
	this->m_features = mp_wrapper->features();
}

inline NoisyErrorFunction::NoisyErrorFunction(NoisyErrorFunction const& op):
	mp_wrapper(op.mp_wrapper->clone())
{
	this->m_features = mp_wrapper->features();
}

inline NoisyErrorFunction& NoisyErrorFunction::operator= (NoisyErrorFunction const& op){
	mp_wrapper.reset(op.mp_wrapper->clone());
	this->m_features = mp_wrapper->features();
	return *this;
}

inline double NoisyErrorFunction::eval(RealVector const& input) const{
	++m_evaluationCounter;
	double value = mp_wrapper->eval(input);
	if(m_regularizer)
		value += m_regularizationStrength * m_regularizer->eval(input);
	return value;
}

inline NoisyErrorFunction::ResultType NoisyErrorFunction::evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative ) const{
	++m_evaluationCounter;
	double value = mp_wrapper->evalDerivative(input,derivative);
	if(m_regularizer){
		FirstOrderDerivative regularizerDerivative;
		value += m_regularizationStrength * m_regularizer->evalDerivative(input,regularizerDerivative);
		noalias(derivative) += m_regularizationStrength*regularizerDerivative;
	}
	return value;
}

}
#endif
