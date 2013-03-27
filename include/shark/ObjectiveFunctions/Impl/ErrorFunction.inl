/*!
 *  \brief implementation of basic error function
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL

#include <boost/cast.hpp>
#include <shark/Core/OpenMP.h>

namespace shark{
namespace detail{

template<class InputType, class LabelType, class OutputType>
class ErrorFunctionWrapper:public FunctionWrapperBase<InputType,LabelType>{
private:
	typedef SupervisedObjectiveFunction<InputType,LabelType> base_type;
public:
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

	void updateFeatures(){
		mep_model->updateFeatures();
		mep_cost->updateFeatures();
		if(mep_model->hasFirstParameterDerivative() && mep_cost->hasFirstDerivative())
			this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		this->m_features|=base_type::CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ErrorFunctionWrapper"; }

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
	
	std::size_t numberOfVariables() const{
		return mep_model->numberOfParameters();
	}
protected:
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractCost<LabelType, OutputType>* mep_cost;
	LabeledData<InputType, LabelType> m_dataset;
};

///\brief Implementation of the Error Function using AbstractCost.
template<class InputType, class LabelType,class OutputType>
class CostBasedErrorFunctionImpl: public ErrorFunctionWrapper<InputType,LabelType,OutputType>{
public:
	
	typedef ErrorFunctionWrapper<InputType,LabelType,OutputType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	CostBasedErrorFunctionImpl(
		AbstractModel<InputType,OutputType>* model, 
		AbstractCost<LabelType, OutputType>* cost
	):base_type(model,cost) {}

	FunctionWrapperBase<InputType,LabelType>* clone()const{
		return new CostBasedErrorFunctionImpl<InputType,LabelType,OutputType>(*this);
	}

	double eval(RealVector const& input) const {
		mep_model->setParameterVector(input);

		Data<OutputType> predictions= (*mep_model)(m_dataset.inputs());
		return mep_cost->eval(m_dataset.labels(), predictions);
	}
	//todo:implement this...
//~ 	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		//~ SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);

		//~ std::size_t dataSize = m_dataset.size();
		//~ mep_model->setParameterVector(point);
		//~ mep_model->resetInternalState();

		//~ derivative.resize(mep_model->numberOfParameters());
		//~ derivative.clear();

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
		//~ for (std::size_t i=0; i != dataSize; i++)
		//~ {
			//~ const InputType& input = m_dataset.input(i);
			//~ mep_model->eval(input);
			//~ calcWeightedDerivative(input,costGradient[i],dataGradient);
			//~ derivative += dataGradient;
		//~ }
		//~ derivative /= dataSize;
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
	using base_type::mep_model;
	using base_type::mep_cost;
	using base_type::m_dataset;
	
};

///\brief Implementation of the ErrorFunction using AbstractLoss.
template<class InputType, class LabelType,class OutputType>
class LossBasedErrorFunctionImpl:public ErrorFunctionWrapper<InputType,LabelType,OutputType>{
public:
	typedef ErrorFunctionWrapper<InputType,LabelType,OutputType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	typedef typename LabeledData<InputType,LabelType>::const_batch_reference const_reference;

	LossBasedErrorFunctionImpl(
		AbstractModel<InputType,OutputType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	):base_type(model,loss), mep_loss(loss) {}

	FunctionWrapperBase<InputType,LabelType>* clone()const{
		return new LossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(*this);
	}

	double eval(RealVector const& input) const {
		mep_model->setParameterVector(input);

		return evalPointSet();
	}
	
	double evalPointSet() const {
		std::size_t dataSize = m_dataset.numberOfElements();

		typename Batch<OutputType>::type prediction;
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			mep_model->eval(batch.input, prediction);
			error += mep_loss->eval(batch.label, prediction);
		}
		return error/dataSize;
	}

	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);
		mep_model->setParameterVector(point);
		return evalDerivativePointSet(derivative);
	}
	
	ResultType evalDerivativePointSet( FirstOrderDerivative & derivative ) const {
		SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);

		std::size_t dataSize = m_dataset.numberOfElements();
		derivative.resize(mep_model->numberOfParameters());
		derivative.clear();

		typename Batch<OutputType>::type prediction;
		RealVector dataGradient(mep_model->numberOfParameters());
		typename Batch<OutputType>::type errorDerivative;

		double error=0.0;
		boost::shared_ptr<State> state = mep_model->createState();
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			// calculate model output for the batch as well as the derivative
			mep_model->eval(batch.input, prediction,*state);

			// calculate error derivative of the loss function
			error += mep_loss->evalDerivative(batch.label, prediction,errorDerivative);

			//calculate the gradient using the chain rule
			mep_model->weightedParameterDerivative(batch.input,errorDerivative,*state,dataGradient);
			derivative+=dataGradient;
		}
		error /= dataSize;
		derivative /= dataSize;
		return error;
	}

protected:
	using base_type::mep_model;
	using base_type::m_dataset;
	AbstractLoss<LabelType, OutputType>* mep_loss;
};


///\brief Implementation of the ErrorFunction using AbstractLoss for parallelizable computations
template<class InputType, class LabelType,class OutputType>
class ParallelLossBasedErrorFunctionImpl:public ErrorFunctionWrapper<InputType,LabelType,OutputType>{
public:
	typedef ErrorFunctionWrapper<InputType,LabelType,OutputType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	typedef typename LabeledData<InputType,LabelType>::const_batch_reference const_reference;

	ParallelLossBasedErrorFunctionImpl(
		AbstractModel<InputType,OutputType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	):base_type(model,loss), mep_loss(loss) {}

	FunctionWrapperBase<InputType,LabelType>* clone()const{
		return new ParallelLossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(*this);
	}

	double eval(RealVector const& input) const {
		SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);
		mep_model->setParameterVector(input);

		std::size_t numBatches = m_dataset.numberOfBatches();
		std::size_t numElements = m_dataset.numberOfElements();
		std::size_t numThreads = std::min(SHARK_NUM_THREADS,numBatches);
		//calculate optimal partitioning
		std::size_t batchesPerThread = numBatches/numThreads;
		std::size_t leftOver = numBatches - batchesPerThread*numThreads;
		double error = 0;
		SHARK_PARALLEL_FOR(int ti = 0; ti < (int)numThreads; ++ti){//MSVC does not support unsigned integrals in paralll loops
			std::size_t t = ti;
			LossBasedErrorFunctionImpl<InputType,LabelType,OutputType> errorFunc(mep_model,mep_loss);
			//get start and end index of batch-range
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			LabeledData<InputType, LabelType> threadData = rangeSubset(m_dataset,start,end);//threadsafe!
			errorFunc.setDataset(threadData);
			errorFunc.setDataset(rangeSubset(m_dataset,start,end));//threadsafe!
			double threadError = errorFunc.evalPointSet();//threadsafe!
			//we need to weight the error and derivativs with the number of samples in the split.
			double weightFactor = double(threadData.numberOfElements())/numElements;
			SHARK_CRITICAL_REGION{
				error += weightFactor * threadError;
			}
		}
		return error;
	}

	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		SHARK_FEATURE_CHECK(HAS_FIRST_DERIVATIVE);
		mep_model->setParameterVector(point);
		derivative.resize(mep_model->numberOfParameters());
		derivative.clear();
		
		std::size_t numBatches = m_dataset.numberOfBatches();
		std::size_t numElements = m_dataset.numberOfElements();
		std::size_t numThreads = std::min(SHARK_NUM_THREADS,numBatches);
		//calculate optimal partitioning
		std::size_t batchesPerThread = numBatches/numThreads;
		std::size_t leftOver = numBatches - batchesPerThread*numThreads;
		double error = 0;
		SHARK_PARALLEL_FOR(int ti = 0; ti < (int)numThreads; ++ti){//MSVC does not support unsigned integrals in paralll loops
			std::size_t t = ti;
			FirstOrderDerivative threadDerivative;
			LossBasedErrorFunctionImpl<InputType,LabelType,OutputType> errorFunc(mep_model,mep_loss);
			//get start and end index of batch-range
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			LabeledData<InputType, LabelType> threadData = rangeSubset(m_dataset,start,end);//threadsafe!
			errorFunc.setDataset(threadData);
			double threadError = errorFunc.evalDerivativePointSet(threadDerivative);//threadsafe!
			//we need to weight the error and derivativs with the number of samples in the split.
			double weightFactor = double(threadData.numberOfElements())/numElements;
			SHARK_CRITICAL_REGION{
				error += weightFactor*threadError;
				noalias(derivative) += weightFactor*threadDerivative;
			}
		}
		return error;
	}

protected:
	using base_type::mep_model;
	using base_type::m_dataset;
	AbstractLoss<LabelType, OutputType>* mep_loss;
};

} // namespace detail


template<class InputType,class LabelType>
void swap(const ErrorFunction<InputType,LabelType>& op1, const ErrorFunction<InputType,LabelType>& op2){
	swap(op1.mp_wrapper,op2.mp_wrapper);
	swap(op1.m_features,op2.m_features);
}

template<class InputType,class LabelType>
template<class OutputType>
ErrorFunction<InputType,LabelType>::ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost){
	
	//check, whether this is a Lossfunction
	if(cost->isLossFunction()){
		AbstractLoss<LabelType, OutputType>* loss = boost::polymorphic_downcast<AbstractLoss<LabelType, OutputType>*>(cost);
		//non squential modls can be parallelized
		if(model->isSequential() || SHARK_NUM_THREADS == 1)
			mp_wrapper.reset(new detail::LossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,loss));
		else
			mp_wrapper.reset(new detail::ParallelLossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,loss));
	}
	else{
		mp_wrapper.reset(new detail::CostBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,cost));
	}
	this -> m_features = mp_wrapper -> features();
}

template<class InputType,class LabelType>
template<class OutputType>
ErrorFunction<InputType,LabelType>::ErrorFunction(AbstractModel<InputType,OutputType>* model, AbstractCost<LabelType, OutputType>* cost, LabeledData<InputType, LabelType> const& dataset){
	//check, whether this is a Lossfunction
	if(cost->isLossFunction()){
		AbstractLoss<LabelType, OutputType>* loss = boost::polymorphic_downcast<AbstractLoss<LabelType, OutputType>*>(cost);
		//non squential modls can be parallelized
		if(model->isSequential() || SHARK_NUM_THREADS == 1)
			mp_wrapper.reset(new detail::LossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,loss));
		else
			mp_wrapper.reset(new detail::ParallelLossBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,loss));
	}
	else{
		mp_wrapper.reset(new detail::CostBasedErrorFunctionImpl<InputType,LabelType,OutputType>(model,cost));
	}
	this -> m_features = mp_wrapper -> features();
	this -> setDataset(dataset);
}

template<class InputType,class LabelType>
ErrorFunction<InputType,LabelType>::ErrorFunction(const ErrorFunction& op)
:mp_wrapper(op.mp_wrapper->clone()){
	this -> m_features = mp_wrapper -> features();
}

template<class InputType,class LabelType>
ErrorFunction<InputType,LabelType>& ErrorFunction<InputType,LabelType>::operator = (const ErrorFunction<InputType,LabelType>& op){
	ErrorFunction<InputType,LabelType> copy(op);
	swap(copy.mp_wrapper,*this);
	return *this;
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::updateFeatures(){
	mp_wrapper -> updateFeatures();
	this -> m_features = mp_wrapper -> features();
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::configure( const PropertyTree & node ){
	mp_wrapper -> configure(node);
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::setDataset(LabeledData<InputType, LabelType> const& dataset){
	mp_wrapper -> setDataset(dataset);
}

template<class InputType,class LabelType>
void ErrorFunction<InputType,LabelType>::proposeStartingPoint(SearchPointType& startingPoint) const{
	mp_wrapper -> proposeStartingPoint(startingPoint);
}

template<class InputType,class LabelType>
std::size_t ErrorFunction<InputType,LabelType>::numberOfVariables() const{
	return mp_wrapper -> numberOfVariables();
}

template<class InputType,class LabelType>
double ErrorFunction<InputType,LabelType>::eval(RealVector const& input) const{
	++(this->m_evaluationCounter);
	return mp_wrapper -> eval(input);
}

template<class InputType,class LabelType>
typename ErrorFunction<InputType,LabelType>::ResultType ErrorFunction<InputType,LabelType>::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const{
	++(this->m_evaluationCounter);
	return mp_wrapper -> evalDerivative(input,derivative);
}

template<class InputType,class LabelType>
typename ErrorFunction<InputType,LabelType>::ResultType ErrorFunction<InputType,LabelType>::evalDerivative( const SearchPointType & input, SecondOrderDerivative & derivative ) const{
	++(this->m_evaluationCounter);
	return mp_wrapper -> evalDerivative(input,derivative);
}
}
#endif
