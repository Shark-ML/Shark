/*!
 *  \brief implementation of basic error function
 *
 *  \author T.Voss, T. Glasmachers, O.Krause
 *  \date 2010-2011
 *
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_ERRORFUNCTION_INL

#include <shark/Core/OpenMP.h>
#include "FunctionWrapperBase.h"

namespace shark{
namespace detail{

///\brief Implementation of the ErrorFunction using AbstractLoss for parallelizable computations
template<class InputType, class LabelType,class OutputType, class SearchPointType>
class ErrorFunctionImpl:public FunctionWrapperBase<SearchPointType>{
private:
	typedef FunctionWrapperBase<SearchPointType> base_type;
public:
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	ErrorFunctionImpl(
		LabeledData<InputType,LabelType> const& dataset,
		AbstractModel<InputType,OutputType, SearchPointType>* model, 
		AbstractLoss<LabelType, OutputType>* loss,
		bool useMiniBatches
	):mep_model(model),mep_loss(loss),m_dataset(dataset), m_useMiniBatches(useMiniBatches){
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);

		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
		this->m_features |= base_type::CAN_PROPOSE_STARTING_POINT;
	}

	std::string name() const
	{ return ""; }

	SearchPointType proposeStartingPoint() const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables() const{
		return mep_model->numberOfParameters();
	}

	ErrorFunctionImpl* clone()const{
		return new ErrorFunctionImpl(*this);
	}

	double eval(SearchPointType const& point) const {
		mep_model->setParameterVector(point);
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			double error = eval(batchIndex,batchIndex+1);
			return error / shark::batchSize(m_dataset.batch(batchIndex));
		}
		
		//full batch case
		std::size_t numBatches = m_dataset.numberOfBatches();
		std::size_t numThreads = std::min(SHARK_NUM_THREADS,numBatches);
		//calculate optimal partitioning
		std::size_t batchesPerThread = numBatches/numThreads;
		std::size_t leftOver = numBatches - batchesPerThread*numThreads;
		double error = 0;
		SHARK_PARALLEL_FOR(int ti = 0; ti < (int)numThreads; ++ti){//MSVC does not support unsigned integrals in parallel loops
			//get start and end index of batch-range
			std::size_t t = ti;
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			
			//compute derivative of the range
			double threadError = eval(start, end);
			
			SHARK_CRITICAL_REGION{
				error +=  threadError;
			}
		}
		return error /  m_dataset.numberOfElements();
	}

	ResultType evalDerivative(SearchPointType const& point, FirstOrderDerivative & derivative ) const {
		mep_model->setParameterVector(point);
		derivative.resize(mep_model->numberOfParameters());
		derivative.clear();
		
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			double error = evalDerivative(batchIndex,batchIndex+1, derivative);
			
			auto const& batch = m_dataset.batch(batchIndex);
			derivative /= shark::batchSize(batch);
			return error / shark::batchSize(batch);
		}
		
		//full batch case
		std::size_t numBatches = m_dataset.numberOfBatches();
		std::size_t numElements = m_dataset.numberOfElements();
		std::size_t numThreads = std::min(SHARK_NUM_THREADS,numBatches);
		//calculate optimal partitioning
		std::size_t batchesPerThread = numBatches/numThreads;
		std::size_t leftOver = numBatches - batchesPerThread*numThreads;
		double error = 0;
		SHARK_PARALLEL_FOR(int ti = 0; ti < (int)numThreads; ++ti){//MSVC does not support unsigned integrals in parallel loops
			//get start and end index of batch-range
			std::size_t t = ti;
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			
			//compute derivative of the range
			FirstOrderDerivative threadDerivative(mep_model->numberOfParameters(),0.0);
			double threadError = evalDerivative(start, end, threadDerivative);
			
			SHARK_CRITICAL_REGION{
				error +=  threadError;
				noalias(derivative) += threadDerivative;
			}
		}
		derivative /= numElements;
		return error / numElements;
	}

protected:
	AbstractModel<InputType, OutputType, SearchPointType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	LabeledData<InputType, LabelType> m_dataset;
	bool m_useMiniBatches;

	ResultType evalDerivative( std::size_t start, std::size_t end,FirstOrderDerivative& derivative) const {
		boost::shared_ptr<State> state = mep_model->createState();
		typename Batch<OutputType>::type predictions;
		typename Batch<OutputType>::type errorDerivative;
		SearchPointType parameterDerivative;
		double errorSum = 0;
		for(std::size_t i = start; i != end; ++i){
			auto const& batch = m_dataset.batch(i);
			mep_model->eval(batch.input,predictions,*state);

			//calculate error derivative of the loss function
			errorSum += mep_loss->evalDerivative(batch.label, predictions,errorDerivative);

			//chain rule
			mep_model->weightedParameterDerivative(batch.input,predictions, errorDerivative,*state,parameterDerivative);
			noalias(derivative) += parameterDerivative;
		}
		return errorSum;
	}
	
	ResultType eval( std::size_t start, std::size_t end) const {
		typename Batch<OutputType>::type predictions;
		double errorSum = 0;
		for(std::size_t i = start; i != end; ++i){
			auto const& batch = m_dataset.batch(i);
			mep_model->eval(batch.input,predictions);

			//calculate error derivative of the loss function
			errorSum += mep_loss->eval(batch.label, predictions);
		}
		return errorSum;
	}
};


///\brief Implementation of the ErrorFunction using AbstractLoss.
template<class InputType, class LabelType,class OutputType, class SearchPointType>
class WeightedErrorFunctionImpl:public FunctionWrapperBase<SearchPointType>{
private:
	typedef FunctionWrapperBase<SearchPointType> base_type;
public:
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;

	WeightedErrorFunctionImpl(
		WeightedLabeledData<InputType, LabelType> const& dataset,
		AbstractModel<InputType,OutputType, SearchPointType>* model, 
		AbstractLoss<LabelType, OutputType>* loss
	):mep_model(model),mep_loss(loss),m_dataset(dataset){
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);

		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
		this-> m_features |= base_type::CAN_PROPOSE_STARTING_POINT;
	}

	std::string name() const
	{ return ""; }

	SearchPointType proposeStartingPoint() const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables() const{
		return mep_model->numberOfParameters();
	}

	WeightedErrorFunctionImpl* clone()const{
		return new WeightedErrorFunctionImpl(*this);
	}

	double eval(SearchPointType const& input) const {
		mep_model->setParameterVector(input);

		double sumWeights = sumOfWeights(m_dataset);
		double error = 0.0;
		SHARK_PARALLEL_FOR(int i = 0; i < (int)m_dataset.numberOfBatches(); ++i){
			auto const& weights = m_dataset.batch(i).weight;
			auto const& data = m_dataset.batch(i).data;
			
			//create model prediction
			auto prediction = (*mep_model)(data.input);
			
			//sum up weighted loss
			double batchError = 0.0;
			for(std::size_t j = 0; j != data.size(); ++j){
				batchError += weights(j) * mep_loss->eval(getBatchElement(data.label,j), getBatchElement(prediction,j));
			}
			SHARK_CRITICAL_REGION{
				error+= batchError;
			}
		}
		return error/sumWeights;
	}

	ResultType evalDerivative( SearchPointType const& point, FirstOrderDerivative& derivative ) const {
		mep_model->setParameterVector(point);
		double sumWeights = sumOfWeights(m_dataset);
		derivative.resize(mep_model->numberOfParameters());
		derivative.clear();
		
		
		double error = 0.0;
		SHARK_PARALLEL_FOR(int i = 0; i < (int)m_dataset.numberOfBatches(); ++i){
			auto const& weights = m_dataset.batch(i).weight;
			auto const& data = m_dataset.batch(i).data;
			
			// calculate model output for the batch as well as the derivative
			typename Batch<OutputType>::type prediction;
			boost::shared_ptr<State> state = mep_model->createState();
			mep_model->eval(data.input, prediction,*state);
			
			//compute  weighted loss and its derivative for every element in its batch
			typename Batch<OutputType>::type errorDerivative(prediction.size1(),prediction.size2());
			OutputType singleDerivative;
			double batchError = 0.0;
			for(std::size_t j = 0; j != data.size(); ++j){
				batchError += weights(j) * mep_loss->evalDerivative(getBatchElement(data.label,j), getBatchElement(prediction,j), singleDerivative);
				noalias(row(errorDerivative,j) ) = weights(j) * singleDerivative;
			}
			
			//calculate the gradient using the chain rule
			SearchPointType dataGradient(mep_model->numberOfParameters());
			mep_model->weightedParameterDerivative(data.input, prediction, errorDerivative,*state,dataGradient);
			SHARK_CRITICAL_REGION{
				derivative += dataGradient;
				error += batchError;
			}
		}
		error /= sumWeights;
		derivative /= sumWeights;
		return error;
	}

private:
	AbstractModel<InputType, OutputType, SearchPointType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	WeightedLabeledData<InputType, LabelType> m_dataset;
};

} // namespace detail
}
#endif
