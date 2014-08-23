/*!
 *  \brief implementation of NoisyErrorFunction
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_NOISYERRORFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_NOISYERRORFUNCTION_H

#include <shark/Data/DataView.h>
#include <shark/Rng/DiscreteUniform.h>

namespace shark{

namespace detail{
/// \brief Implementation for the NoisyErrorFunction. It hides the Type of the OutputType.
template<class InputType,class LabelType, class OutputType>
class NoisyErrorFunctionWrapper : public NoisyErrorFunctionWrapperBase
{
private:
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractLoss<LabelType,OutputType>* mep_loss;
	DataView<LabeledData<InputType,LabelType> const> m_dataset;
	unsigned int m_batchSize;
	mutable DiscreteUniform<Rng::rng_type> m_uni;
	typedef typename AbstractModel<InputType, OutputType>::BatchOutputType BatchOutputType;
	typedef typename LabeledData<InputType,LabelType>::batch_type BatchDataType;

public:
	NoisyErrorFunctionWrapper(
		LabeledData<InputType,LabelType> const& dataset,
		AbstractModel<InputType,OutputType>* model,
		AbstractLoss<LabelType,OutputType>* loss,unsigned int batchSize=1
	): mep_model(model), mep_loss(loss), m_dataset(dataset)
	, m_batchSize(batchSize),m_uni(Rng::globalRng,0,m_dataset.size()-1)
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

	void configure( const PropertyTree & node ) {
		PropertyTree::const_assoc_iterator it = node.find("model");
		if(it!=node.not_found())
		{
			mep_model->configure(it->second);
		}
		it = node.find("loss");
		if(it!=node.not_found())
		{
			mep_loss->configure(it->second);
		}
		m_batchSize = node.get("batchSize",1);

		updateFeatures();
	}

	void proposeStartingPoint( SearchPointType & startingPoint)const {
		startingPoint=mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	double eval(const RealVector & input)const {
		mep_model->setParameterVector(input);
		
		//prepare batch for the current iteration
		std::vector<std::size_t> indices(m_batchSize);
		std::generate(indices.begin(),indices.end(),m_uni);
		BatchDataType  batch = subBatch(m_dataset,indices);
		
		BatchOutputType predictions;
		mep_model->eval(batch.input,predictions);

		//calculate error derivative of the loss function
		double error= mep_loss->eval(batch.label, predictions);
		error /= m_batchSize;
		return error;
	}

	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative )const {
		mep_model->setParameterVector(input);
		boost::shared_ptr<State> state = mep_model->createState();
		
		//prepare batch for the current iteration
		std::vector<std::size_t> indices(m_batchSize);
		std::generate(indices.begin(),indices.end(),m_uni);
		BatchDataType  batch = subBatch(m_dataset,indices);
		
		BatchOutputType predictions;
		mep_model->eval(batch.input,predictions,*state);

		//calculate error derivative of the loss function
		BatchOutputType errorDerivative;
		double error= mep_loss->evalDerivative(batch.label, predictions,errorDerivative);

		//chain rule
		mep_model->weightedParameterDerivative(batch.input,errorDerivative,*state,derivative);
	
		error/=m_batchSize;
		derivative/= m_batchSize;
		return error;
	}
};
}
template<class InputType,class LabelType>
void swap(const NoisyErrorFunction<InputType,LabelType>& op1, const NoisyErrorFunction<InputType,LabelType>& op2){
	swap(op1.mp_wrapper,op2.mp_wrapper);
	swap(op1.m_features,op2.m_features);
}

template<class InputType,class LabelType>
template<class OutputType>
NoisyErrorFunction<InputType,LabelType>::NoisyErrorFunction(
	DatasetType const& dataset,
	AbstractModel<InputType,OutputType>* model,
	AbstractLoss<LabelType, OutputType>* loss,
	unsigned int batchSize)
:mp_wrapper(new detail::NoisyErrorFunctionWrapper<InputType,LabelType,OutputType>(dataset,model,loss,batchSize))
, m_regularizer(0), m_regularizationStrength(0){
	this -> m_features = mp_wrapper -> features();
}
template<class InputType,class LabelType>
NoisyErrorFunction<InputType,LabelType>::NoisyErrorFunction(const NoisyErrorFunction<InputType,LabelType>& op){
	mp_wrapper = op.mp_wrapper->clone();
	this -> m_features = mp_wrapper -> features();
}

template<class InputType,class LabelType>
NoisyErrorFunction<InputType,LabelType>& 
NoisyErrorFunction<InputType,LabelType>::operator = (const NoisyErrorFunction<InputType,LabelType>& op){
	NoisyErrorFunction<InputType,LabelType> copy(op);
	swap(copy,*this);
	return *this;
}

template<class InputType,class LabelType>
void NoisyErrorFunction<InputType,LabelType>::updateFeatures(){
	mp_wrapper -> updateFeatures();
	this -> m_features = mp_wrapper -> features();
}

template<class InputType,class LabelType>
void NoisyErrorFunction<InputType,LabelType>::configure( const PropertyTree & node ){
	mp_wrapper -> configure(node);
}

template<class InputType,class LabelType>
void NoisyErrorFunction<InputType,LabelType>::proposeStartingPoint(SearchPointType& startingPoint) const{
	mp_wrapper -> proposeStartingPoint(startingPoint);
}

template<class InputType,class LabelType>
std::size_t NoisyErrorFunction<InputType,LabelType>::numberOfVariables() const{
	return mp_wrapper -> numberOfVariables();
}

template<class InputType,class LabelType>
double NoisyErrorFunction<InputType,LabelType>::eval(RealVector const& input) const{
	++m_evaluationCounter;
	double value = mp_wrapper -> eval(input);
	if(m_regularizer)
		value += m_regularizationStrength * m_regularizer->eval(input);
	return value;
}
template<class InputType,class LabelType>
typename NoisyErrorFunction<InputType,LabelType>::ResultType 
NoisyErrorFunction<InputType,LabelType>::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const{
	++m_evaluationCounter;
	double value = mp_wrapper -> evalDerivative(input,derivative);
	if(m_regularizer){
		FirstOrderDerivative regularizerDerivative;
		value += m_regularizationStrength * m_regularizer->evalDerivative(input,regularizerDerivative);
		noalias(derivative) += m_regularizationStrength*regularizerDerivative;
	}
	return value;
}

template<class InputType,class LabelType>
void NoisyErrorFunction<InputType,LabelType>::setBatchSize(unsigned int batchSize){
	mp_wrapper -> setBatchSize(batchSize);
}
template<class InputType,class LabelType>
unsigned int NoisyErrorFunction<InputType,LabelType>::batchSize() const{
	return mp_wrapper -> batchSize();
}

}
#endif
