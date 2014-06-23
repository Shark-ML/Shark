/*!
 * 
 *
 * \brief       Negative Log Likelihood error function
 * 
 * 
 *
 * \author      O.Krause
 * \date        2014
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_LOG_LIKELIHOOD_H
#define SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_LOG_LIKELIHOOD_H

#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
namespace shark{

/// \brief Computes the negative log likelihood of a dataset under a model
///
/// The negative log likelihood is defined as 
/// \f[ L(\theta) = -\frac 1 N \sum_{i=1}^N log(p_{\theta}(x_i)) \f]
/// where \f$ \theta \f$ is the vector of parameters of the model \f$ p \f$ and \f$ x \f$ are the
/// datapoints of the training set. Minimizing this
/// maximizes the probability of the datast under p. This error measure is
/// closely related to the Kulback-Leibler-Divergence.
///
/// For this error function, the model is only allowed to have a single output
/// - the probability of the sample. The distribution must be normalized as otherwise
/// the likeelihood does not mean anything. 
class NegativeLogLikelihood : public UnsupervisedObjectiveFunction<RealVector>
{
public:
	NegativeLogLikelihood(
		AbstractModel<RealVector,RealVector>* model
	):mep_model(model){
		if(mep_model->hasFirstParameterDerivative())
			m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}
	NegativeLogLikelihood(
		AbstractModel<RealVector,RealVector>* model,
		UnlabeledData<RealVector> const& data
	):mep_model(model),m_data(data){
		if(mep_model->hasFirstParameterDerivative())
			m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeLogLikelihood"; }

	
	void configure(PropertyTree const& node){}
	void setData(UnlabeledData<RealVector> const& data){
		m_data = data;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint=mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	ResultType eval(RealVector const& input) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		m_evaluationCounter++;
		mep_model->setParameterVector(input);
		
		double error = 0;
		double minProb = 1e-100;//numerical stability is only guaranteed for lower bounded probabilities
		SHARK_PARALLEL_FOR(int i = 0; i < (int)m_data.numberOfBatches(); ++i){
			RealMatrix predictions = (*mep_model)(m_data.batch(i));
			SIZE_CHECK(predictions.size2() == 1);
			double logLikelihoodOfSamples = sum(log(max(predictions,minProb)));
			SHARK_CRITICAL_REGION{
				error += logLikelihoodOfSamples;
			}
		}
		error/=m_data.numberOfElements();//compute mean
		return -error;//negative log likelihood
	}
	ResultType evalDerivative( 
		SearchPointType const& input, 
		FirstOrderDerivative & derivative 
	) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		m_evaluationCounter++;
		mep_model->setParameterVector(input);
		derivative.resize(input.size());
		derivative.clear();
		
		//compute partitioning on threads
		std::size_t numBatches = m_data.numberOfBatches();
		std::size_t numElements = m_data.numberOfElements();
		std::size_t numThreads = std::min(SHARK_NUM_THREADS,numBatches);
		//calculate optimal partitioning
		std::size_t batchesPerThread = numBatches/numThreads;
		std::size_t leftOver = numBatches - batchesPerThread*numThreads;
		double error = 0;
		double minProb = 1e-100;//numerical stability is only guaranteed for lower bounded probabilities
		SHARK_PARALLEL_FOR(int ti = 0; ti < (int)numThreads; ++ti){//MSVC does not support unsigned integrals in paralll loops
			std::size_t t = ti;
			//~ //get start and end index of batch-range
			std::size_t start = t*batchesPerThread+std::min(t,leftOver);
			std::size_t end = (t+1)*batchesPerThread+std::min(t+1,leftOver);
			
			//calculate error and derivative of the current thread
			FirstOrderDerivative threadDerivative(input.size(),0.0);
			double threadError = 0;
			boost::shared_ptr<State> state = mep_model->createState();
			RealVector batchDerivative;
			RealMatrix predictions;
			for(std::size_t i  = start; i != end; ++i){
				mep_model->eval(m_data.batch(i),predictions,*state);
				SIZE_CHECK(predictions.size2() == 1);
				threadError += sum(log(max(predictions,minProb)));
				//noalias(predictions) = elem_inv(predictions)
				//the below handls numeric instabilities...
				for(std::size_t j = 0; j != predictions.size1(); ++j){
					for(std::size_t k = 0; k != predictions.size2(); ++k){
						if(predictions(j,k) < minProb){
							predictions(j,k) = 0;
						}
						else{
							predictions(j,k) = 1.0/predictions(j,k);
						}
					}
				}
				mep_model->weightedParameterDerivative(
					m_data.batch(i),predictions,*state,batchDerivative
				);
				threadDerivative += batchDerivative;
			}
			
			//sum over all threads
			SHARK_CRITICAL_REGION{
				error += threadError;
				noalias(derivative) += threadDerivative;
			}
		}
		
		error /= numElements;
		derivative /= numElements;
		derivative *= -1;
		return -error;//negative log likelihood
	}

private:
	AbstractModel<RealVector,RealVector>* mep_model;
	UnlabeledData<RealVector> m_data;
};

}
#endif
