/*!
 *  \brief implementation of basic error function
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL

#include <boost/cast.hpp>

namespace shark{
namespace detail{


///\brief Implementation of the Error Function using AbstractCost.
template<class InputType, class LabelType,class OutputType>
class ErrorFunctionWrapper:public FunctionWrapperBase<InputType,LabelType>{
public:
	typedef SupervisedObjectiveFunction<InputType,LabelType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	ErrorFunctionWrapper(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost) {
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(cost!=NULL);
		mep_model = model;
		mep_cost = cost;
		updateFeatures();
	}

	FunctionWrapperBase<InputType,LabelType>* clone()const{
		return new ErrorFunctionWrapper<InputType,LabelType,OutputType>(*this);
	}

	void updateFeatures(){
		mep_model->updateFeatures();
		mep_cost->updateFeatures();
		if(mep_model->hasFirstParameterDerivative() && mep_cost->hasFirstDerivative())
			this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		this->m_features|=base_type::CAN_PROPOSE_STARTING_POINT;
		//a hack to update changing names...
		this->m_name="ErrorFunction<"+mep_model->name()+","+mep_cost->name()+">";
	}

	void configure( const PropertyTree & node ) {
		PropertyTree::const_assoc_iterator it = node.find("model");
		if(it!=node.not_found())
		{
			mep_model->configure(it->second);
		}
		// be flexible; allow for "cost" or "loss"
		it = node.find("cost");
		if(it!=node.not_found())
		{
			mep_cost->configure(it->second);
		}
		it = node.find("loss");
		if(it!=node.not_found())
		{
			mep_cost->configure(it->second);
		}
		updateFeatures();
	}

	void setDataset(LabeledData<InputType, LabelType> const& dataset){
		m_dataset = dataset;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mep_model->parameterVector();
	}

	double eval(RealVector const& input) const {
		mep_model->setParameterVector(input);

		Data<OutputType> predictions= (*mep_model)(m_dataset.inputs());
		return mep_cost->eval(m_dataset.labels(), predictions);
	}
	//todo:implement this...
//~ 	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		//~ SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);

		//~ size_t dataSize = m_dataset.size();
		//~ mep_model->setParameterVector(point);
		//~ mep_model->resetInternalState();

		//~ derivative.m_gradient.resize(mep_model->numberOfParameters());
		//~ derivative.m_gradient.clear();

		//~ // General computation for non-separable cost functions.
		//~ // This is not optimally efficient, because the model
		//~ // prediction needs to be computed twice - once because
		//~ // we need the value, and the second time to prepare the
		//~ // computation of the weighted derivative.
		//~ RealVector dataGradient;
		//~ std::vector<OutputType> costGradient;
		//~ Data<OutputType> predictions(dataSize);
		//~ mep_model->eval(m_dataset.inputs(), predictions);
		//~ double error = mep_cost->evalDerivative(m_dataset.labels(), predictions, costGradient);
		//~ for (size_t i=0; i != dataSize; i++)
		//~ {
			//~ const InputType& input = m_dataset.input(i);
			//~ mep_model->eval(input);
			//~ calcWeightedDerivative(input,costGradient[i],dataGradient);
			//~ derivative.m_gradient += dataGradient;
		//~ }
		//~ derivative.m_gradient /= dataSize;
		//~ return error;
//~ 		return 0;
//~ 	}

private:
	
	template<class T,class Vector>
	void calcWeightedDerivative(const InputType& input, const T& weights, Vector& inputGradient)const{
		mep_model->weightedParameterDerivative(input, weights, inputGradient);
	}
	//dummy for classification
	template<class Vector>
	void calcWeightedDerivative(const InputType& input, int weights, Vector& inputGradient)const{
		(void)input;
		(void)weights;
		(void)inputGradient;
	}
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractCost<LabelType, OutputType>* mep_cost;
	LabeledData<InputType, LabelType> m_dataset;
};

///\brief Implementation of the ErrorFunction using AbstractLoss.
template<class InputType, class LabelType,class OutputType>
class LossBasedErrorFunctionWrapper:public FunctionWrapperBase<InputType,LabelType>{
public:
	typedef SupervisedObjectiveFunction<InputType,LabelType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	typedef typename LabeledData<InputType,LabelType>::const_reference const_reference;

	LossBasedErrorFunctionWrapper(AbstractModel<InputType,OutputType>* model, AbstractLoss<LabelType, OutputType>* loss) {
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);
		mep_model = model;
		mep_loss = loss;
		updateFeatures();
	}

	FunctionWrapperBase<InputType,LabelType>* clone()const{
		return new LossBasedErrorFunctionWrapper<InputType,LabelType,OutputType>(*this);
	}

	void updateFeatures(){
		mep_model->updateFeatures();
		mep_loss->updateFeatures();
		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		//~ if(mep_model->hasSecondParameterDerivative() && mep_loss->hasSecondDerivative())
			//~ this->m_features|=base_type::HAS_SECOND_DERIVATIVE;
		this->m_features|=base_type::CAN_PROPOSE_STARTING_POINT;
		//a hack to update changing names...
		this->m_name="ErrorFunction<"+mep_model->name()+","+mep_loss->name()+">";
	}

	void configure( const PropertyTree & node ) {
		PropertyTree::const_assoc_iterator it = node.find("model");
		if(it!=node.not_found())
		{
			mep_model->configure(it->second);
		}
		// be flexible; allow for "cost" or "loss"
		it = node.find("cost");
		if(it!=node.not_found())
		{
			mep_loss->configure(it->second);
		}
		it = node.find("loss");
		if(it!=node.not_found())
		{
			mep_loss->configure(it->second);
		}
		updateFeatures();
	}

	void setDataset(LabeledData<InputType, LabelType> const& dataset){
		m_dataset = dataset;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mep_model->parameterVector();
	}

	double eval(RealVector const& input) const {
		size_t dataSize = m_dataset.numberOfElements();
		mep_model->setParameterVector(input);

		typename Batch<OutputType>::type prediction;
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset){
			mep_model->eval(batch.input, prediction);
			error += mep_loss->eval(batch.label, prediction);
		}
		error /= (double)dataSize;
		return error;
	}

	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);

		size_t dataSize = m_dataset.numberOfElements();
		mep_model->setParameterVector(point);

		derivative.m_gradient.resize(mep_model->numberOfParameters());
		derivative.m_gradient.clear();

		typename Batch<OutputType>::type prediction;
		RealVector dataGradient(mep_model->numberOfParameters());
		typename Batch<OutputType>::type errorDerivative;

		double error=0.0;
		boost::shared_ptr<State> state = mep_model->createState();
		BOOST_FOREACH(const_reference batch,m_dataset){
			// calculate model output for the batch as well as the derivative
			mep_model->eval(batch.input, prediction,*state);

			// calculate error derivative of the loss function
			error += mep_loss->evalDerivative(batch.label, prediction,errorDerivative);

			//calculate the gradient using the chain rule
			mep_model->weightedParameterDerivative(batch.input,errorDerivative,*state,dataGradient);
			derivative.m_gradient+=dataGradient;
		}
		error /= dataSize;
		derivative.m_gradient /= dataSize;
		return error;
	}

//~ 	ResultType evalDerivative( const SearchPointType & point, SecondOrderDerivative & derivative)const {
		//~ SHARK_FEATURE_CHECK(HAS_SECOND_DERIVATIVE);

		//~ size_t dataSize = m_dataset.size();
		//~ mep_model->setParameterVector(point);
		//~ mep_model->resetInternalState();

		//~ size_t parameters=mep_model->numberOfParameters();

		//~ derivative.m_gradient.resize(parameters);
		//~ derivative.m_hessian.resize(parameters,parameters);
		//~ derivative.m_gradient.clear();
		//~ derivative.m_hessian.clear();

		//~ OutputType prediction;
		//~ OutputType errorDerivative;
		//~ RealMatrix errorHessian;

		//~ RealVector dataGradient;
		//~ RealMatrix dataHessian;

		//~ double error=0;
		//~ for (size_t i=0; i<dataSize; i++) {
			//~ //calculate model output for one single input as well as the derivative
			//~ const InputType& input = m_dataset.input(i);
			//~ mep_model->eval(input, prediction);

			//~ // calculate error derivative and Hessian of the loss function
			//~ error += mep_loss->evalDerivative(m_dataset.label(i), prediction,errorDerivative,errorHessian);

			//~ mep_model->weightedParameterDerivative(input,errorDerivative,errorHessian,dataGradient,dataHessian);

			//~ derivative.m_gradient += dataGradient;
			//~ derivative.m_hessian += dataHessian;
		//~ }
		//~ error /= dataSize;
		//~ derivative.m_gradient/=dataSize;
		//~ derivative.m_hessian/=dataSize;
		//~ return error;
//~ 	}

protected:
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	LabeledData<InputType, LabelType> m_dataset;
};


} // namespace detail


template<class InputType,class LabelType>
void swap(const ErrorFunction<InputType,LabelType>& op1, const ErrorFunction<InputType,LabelType>& op2){
	swap(op1.m_wrapper,op2.m_wrapper);
	swap(op1.m_name,op2.m_name);
	swap(op1.m_features,op2.m_features);
}

template<class InputType,class LabelType>
template<class OutputType>
ErrorFunction<InputType,LabelType>::ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost){
	//check, whether this is a Lossfunction
	
	if(cost->isLossFunction()){
		AbstractLoss<LabelType, OutputType>* loss = boost::polymorphic_downcast<AbstractLoss<LabelType, OutputType>*>(cost);
		m_wrapper.reset(new detail::LossBasedErrorFunctionWrapper<InputType,LabelType,OutputType>(model,loss));
	}
	else{
		m_wrapper.reset(new detail::ErrorFunctionWrapper<InputType,LabelType,OutputType>(model,cost));
	}
	this -> m_name = m_wrapper->name();
	this -> m_features = m_wrapper -> features();
}

template<class InputType,class LabelType>
template<class OutputType>
ErrorFunction<InputType,LabelType>::ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost, LabeledData<InputType, LabelType> const& dataset){
	//check, whether this is a Lossfunction
	if(cost->isLossFunction()){
		AbstractLoss<LabelType, OutputType>* loss = boost::polymorphic_downcast<AbstractLoss<LabelType, OutputType>*>(cost);
		m_wrapper.reset(new detail::LossBasedErrorFunctionWrapper<InputType,LabelType,OutputType>(model,loss));
	}
	else{
		m_wrapper.reset(new detail::ErrorFunctionWrapper<InputType,LabelType,OutputType>(model,cost));
	}
	this -> m_name = m_wrapper->name();
	this -> m_features = m_wrapper -> features();

	this -> setDataset(dataset);
}

template<class InputType,class LabelType>
ErrorFunction<InputType,LabelType>::ErrorFunction(const ErrorFunction& op)
:m_wrapper(op.m_wrapper->clone()){
	this -> m_name = m_wrapper -> name();
	this -> m_features = m_wrapper -> features();
}

template<class InputType,class LabelType>
ErrorFunction<InputType,LabelType>& ErrorFunction<InputType,LabelType>::operator = (const ErrorFunction<InputType,LabelType>& op){
	ErrorFunction<InputType,LabelType> copy(op);
	swap(copy.m_wrapper,*this);
	return *this;
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::updateFeatures(){
	m_wrapper -> updateFeatures();
	this -> m_name = m_wrapper -> name();
	this -> m_features = m_wrapper -> features();
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::configure( const PropertyTree & node ){
	m_wrapper -> configure(node);
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::setDataset(LabeledData<InputType, LabelType> const& dataset){
	m_wrapper -> setDataset(dataset);
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::proposeStartingPoint(SearchPointType& startingPoint) const{
	m_wrapper -> proposeStartingPoint(startingPoint);
}

template<class InputType,class LabelType>
double ErrorFunction<InputType,LabelType>::eval(RealVector const& input) const{
	++(this->m_evaluationCounter);
	return m_wrapper -> eval(input);
}

template<class InputType,class LabelType>
typename ErrorFunction<InputType,LabelType>::ResultType ErrorFunction<InputType,LabelType>::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const{
	++(this->m_evaluationCounter);
	return m_wrapper -> evalDerivative(input,derivative);
}

template<class InputType,class LabelType>
typename ErrorFunction<InputType,LabelType>::ResultType ErrorFunction<InputType,LabelType>::evalDerivative( const SearchPointType & input, SecondOrderDerivative & derivative ) const{
	++(this->m_evaluationCounter);
	return m_wrapper -> evalDerivative(input,derivative);
}
}
#endif
