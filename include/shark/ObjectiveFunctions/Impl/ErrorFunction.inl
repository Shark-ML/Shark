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

#include <shark/Core/Threading/Algorithms.h>
#include "FunctionWrapperBase.h"

namespace shark{
namespace detail{

///\brief Implementation of the ErrorFunction using AbstractLoss for parallelizable computations
template<class InputType, class LabelType,class OutputType, class SearchPointType>
class ErrorFunctionImpl:public FunctionWrapperBase<SearchPointType>{
private:
	typedef FunctionWrapperBase<SearchPointType> base_type;
	typedef typename LabeledData<InputType, LabelType>::const_batch_reference batch_reference;
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
		
		auto processBatch = [&](batch_reference batch){
			typename Batch<OutputType>::type predictions;
			mep_model->eval(batch.input,predictions);
			return mep_loss->eval(batch.label, predictions);
		};
		
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			double error = processBatch(m_dataset.batch(batchIndex));
			return error / shark::batchSize(m_dataset.batch(batchIndex));
		}
		
		//full batch case
		double error = threading::mapAccumulate(m_dataset.batches(), 0.0, processBatch, threading::globalThreadPool());
		return error/m_dataset.numberOfElements();
	}

	ResultType evalDerivative(SearchPointType const& point, FirstOrderDerivative & derivative ) const {
		mep_model->setParameterVector(point);
		
		//compute error and derivative of single batch
		typedef std::pair<double, FirstOrderDerivative> batch_result;
		auto processBatch = [&](batch_reference batch){
			boost::shared_ptr<State> state = mep_model->createState();
			typename Batch<OutputType>::type predictions;
			mep_model->eval(batch.input,predictions,*state);

			//calculate error derivative of the loss function
			typename Batch<OutputType>::type errorDerivative;
			double error = mep_loss->evalDerivative(batch.label, predictions,errorDerivative);

			//chain rule
			SearchPointType parameterDerivative;
			mep_model->weightedParameterDerivative(batch.input,predictions, errorDerivative,*state,parameterDerivative);
			return batch_result(error, std::move(parameterDerivative));
		};
		
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			batch_result result = processBatch(m_dataset.batch(batchIndex));
			
			derivative = std::move(result.second);
			auto const& batch = m_dataset.batch(batchIndex);
			derivative /= shark::batchSize(batch);
			return result.first / shark::batchSize(batch);
		}
		
		//full batch case
		//sums up error and derivatives of different batches
		auto sumBatches=[](batch_result acc, batch_result const& batch){
			acc.first += batch.first;
			acc.second += batch.second;
			return std::move(acc);
		};
		//compute the derivative in parallel
		batch_result result = threading::mapReduce(
			m_dataset.batches(),
			batch_result(0.0,FirstOrderDerivative(mep_model->numberOfParameters(), 0.0)),
			processBatch, sumBatches,
			threading::globalThreadPool()
		);
		
		double sumWeights = (double)m_dataset.numberOfElements();
		derivative = result.second; 
		derivative /= sumWeights;
		return result.first / sumWeights;
	}

protected:
	AbstractModel<InputType, OutputType, SearchPointType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	LabeledData<InputType, LabelType> m_dataset;
	bool m_useMiniBatches;
};


///\brief Implementation of the ErrorFunction using AbstractLoss.
template<class InputType, class LabelType,class OutputType, class SearchPointType>
class WeightedErrorFunctionImpl:public FunctionWrapperBase<SearchPointType>{
private:
	typedef FunctionWrapperBase<SearchPointType> base_type;
	typedef std::pair<double, SearchPointType> batch_result;
	typedef typename WeightedLabeledData<InputType, LabelType>::const_batch_reference batch_reference;
public:
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;

	WeightedErrorFunctionImpl(
		WeightedLabeledData<InputType, LabelType> const& dataset,
		AbstractModel<InputType,OutputType, SearchPointType>* model, 
		AbstractLoss<LabelType, OutputType>* loss,
		bool useMiniBatches
	):mep_model(model),mep_loss(loss),m_dataset(dataset), m_useMiniBatches(useMiniBatches){
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

		
		//computes error of a single batch
		auto processBatch = [&](batch_reference batch){
			auto const& weights = batch.weight;
			auto const& data = batch.data;
			
			//create model prediction
			auto prediction = (*mep_model)(data.input);
			
			//sum up weighted loss
			double batchError = 0.0;
			for(std::size_t j = 0; j != data.size(); ++j){
				batchError += weights(j) * mep_loss->eval(getBatchElement(data.label,j), getBatchElement(prediction,j));
			}
			return batchError;
		};
		
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			double error = processBatch(m_dataset.batch(batchIndex));
			return error / sum(m_dataset.batch(batchIndex).weight);
		}
		
		double error = threading::mapAccumulate(m_dataset.batches(), 0.0, processBatch, threading::globalThreadPool());
		double sumWeights = sumOfWeights(m_dataset);
		return error/sumWeights;
	}

	ResultType evalDerivative( SearchPointType const& point, FirstOrderDerivative& derivative ) const {
		mep_model->setParameterVector(point);

		//computes error and derivative of a single batch
		typedef std::pair<double, FirstOrderDerivative> batch_result;
		auto processBatch = [&](batch_reference batch){
			auto const& weights = batch.weight;
			auto const& data = batch.data;
			
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
			};
			
			//calculate the gradient using the chain rule
			SearchPointType batchGradient(mep_model->numberOfParameters());
			mep_model->weightedParameterDerivative(data.input, prediction, errorDerivative,*state,batchGradient);
			
			return batch_result(batchError, std::move(batchGradient));
		};
		
		//minibatch case
		if(m_useMiniBatches){
			std::size_t batchIndex = random::discrete(*this->mep_rng, std::size_t(0),m_dataset.numberOfBatches()-1);
			auto const& batch = m_dataset.batch(batchIndex);
			batch_result result = processBatch(batch);
			
			derivative = std::move(result.second);
			derivative /= sum(batch.weight);
			return result.first / sum(batch.weight);
		}
		
		
		//sums up error and derivatives of different batches
		auto sumBatches=[](batch_result acc, batch_result const& batch){
			acc.first += batch.first;
			acc.second += batch.second;
			return std::move(acc);
		};
		//compute the derivative in parallel
		batch_result result = threading::mapReduce(
			m_dataset.batches(),
			batch_result(0.0,FirstOrderDerivative(mep_model->numberOfParameters(), 0.0)),
			processBatch, sumBatches,
			threading::globalThreadPool()
		);
		
		double sumWeights = sumOfWeights(m_dataset);
		derivative = result.second; 
		derivative /= sumWeights;
		return result.first / sumWeights;
	}

private:
	AbstractModel<InputType, OutputType, SearchPointType>* mep_model;
	AbstractLoss<LabelType, OutputType>* mep_loss;
	WeightedLabeledData<InputType, LabelType> m_dataset;
	bool m_useMiniBatches;
};

} // namespace detail
}
#endif
