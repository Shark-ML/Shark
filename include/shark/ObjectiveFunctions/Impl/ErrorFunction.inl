/*!
 *  \brief implementation of basic error function
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL

#include <shark/Core/OpenMP.h>

namespace shark{
namespace detail{


///\brief Implementation of the ErrorFunction using AbstractLoss.
template<class InputType, class LabelType,class OutputType>
class ErrorFunctionImpl:public FunctionWrapperBase{
public:
	ErrorFunctionImpl(
		LabeledData<InputType, LabelType> const& dataset,
		AbstractModel<InputType,OutputType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	):mep_model(model),mep_loss(loss),m_dataset(dataset){
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);

		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			m_features|=HAS_FIRST_DERIVATIVE;
		m_features|=CAN_PROPOSE_STARTING_POINT;
	}

	std::string name() const
	{ return ""; }

	SearchPointType proposeStartingPoint() const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables() const{
		return mep_model->numberOfParameters();
	}

	FunctionWrapperBase* clone()const{
		return new ErrorFunctionImpl<InputType,LabelType,OutputType>(*this);
	}

	double eval(RealVector const& input) const {
		mep_model->setParameterVector(input);

		return evalPointSet();
	}
	
	double evalPointSet() const {
		std::size_t dataSize = m_dataset.numberOfElements();
		typedef typename LabeledData<InputType,LabelType>::const_batch_reference const_reference;
		
		typename Batch<OutputType>::type prediction;
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			mep_model->eval(batch.input, prediction);
			error += mep_loss->eval(batch.label, prediction);
		}
		return error/dataSize;
	}

	ResultType evalDerivative( const SearchPointType & point, FirstOrderDerivative & derivative ) const {
		mep_model->setParameterVector(point);
		return evalDerivativePointSet(derivative);
	}
	
	ResultType evalDerivativePointSet( FirstOrderDerivative & derivative ) const {
		typedef typename LabeledData<InputType,LabelType>::const_batch_reference const_reference;
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
		derivative /= double(dataSize);
		return error;
	}

private:
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	LabeledData<InputType, LabelType> m_dataset;
};


///\brief Implementation of the ErrorFunction using AbstractLoss for parallelizable computations
template<class InputType, class LabelType,class OutputType>
class ParallelErrorFunctionImpl:public FunctionWrapperBase{
public:

	ParallelErrorFunctionImpl(
		LabeledData<InputType,LabelType> const& dataset,
		AbstractModel<InputType,OutputType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	):mep_model(model),mep_loss(loss),m_dataset(dataset){
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);

		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			m_features|=HAS_FIRST_DERIVATIVE;
		m_features|=CAN_PROPOSE_STARTING_POINT;
	}

	std::string name() const
	{ return ""; }

	SearchPointType proposeStartingPoint() const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables() const{
		return mep_model->numberOfParameters();
	}

	FunctionWrapperBase* clone()const{
		return new ParallelErrorFunctionImpl<InputType,LabelType,OutputType>(*this);
	}

	double eval(RealVector const& input) const {
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
			//get start and end index of batch-range
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			LabeledData<InputType, LabelType> threadData = rangeSubset(m_dataset,start,end);//threadsafe!
			ErrorFunctionImpl<InputType,LabelType,OutputType> errorFunc(threadData,mep_model,mep_loss);
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
			//get start and end index of batch-range
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			LabeledData<InputType, LabelType> threadData = rangeSubset(m_dataset,start,end);//threadsafe!
			ErrorFunctionImpl<InputType,LabelType,OutputType> errorFunc(threadData,mep_model,mep_loss);
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
	AbstractModel<InputType, OutputType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	LabeledData<InputType, LabelType> m_dataset;
};

} // namespace detail


inline void swap(ErrorFunction& op1, ErrorFunction& op2){
	using std::swap;
	swap(op1.mp_wrapper,op2.mp_wrapper);
	swap(op1.m_features,op2.m_features);
}

template<class InputType,class LabelType, class OutputType>
inline ErrorFunction::ErrorFunction(
	LabeledData<InputType, LabelType> const& dataset,
	AbstractModel<InputType,OutputType>* model, 
	AbstractLoss<LabelType, OutputType>* loss
){
	m_regularizer = 0;
	//non sequential models can be parallelized
	if(model->isSequential() || SHARK_NUM_THREADS == 1)
		mp_wrapper.reset(new detail::ErrorFunctionImpl<InputType,LabelType,OutputType>(dataset,model,loss));
	else
		mp_wrapper.reset(new detail::ParallelErrorFunctionImpl<InputType,LabelType,OutputType>(dataset,model,loss));

	this -> m_features = mp_wrapper -> features();
}

inline ErrorFunction::ErrorFunction(const ErrorFunction& op)
:mp_wrapper(op.mp_wrapper->clone()){
	this -> m_features = mp_wrapper -> features();
}

inline ErrorFunction& ErrorFunction::operator = (const ErrorFunction& op){
	ErrorFunction copy(op);
	swap(copy.mp_wrapper,mp_wrapper);
	return *this;
}

inline double ErrorFunction::eval(RealVector const& input) const{
	++m_evaluationCounter;
	double value = mp_wrapper -> eval(input);
	if(m_regularizer)
		value += m_regularizationStrength * m_regularizer->eval(input);
	return value;
}

inline ErrorFunction::ResultType ErrorFunction::evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const{
	++m_evaluationCounter;
	double value = mp_wrapper -> evalDerivative(input,derivative);
	if(m_regularizer){
		FirstOrderDerivative regularizerDerivative;
		value += m_regularizationStrength * m_regularizer->evalDerivative(input,regularizerDerivative);
		noalias(derivative) += m_regularizationStrength*regularizerDerivative;
	}
	return value;
}
}
#endif
