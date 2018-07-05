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
#ifndef SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_LOG_LIKELIHOOD_H
#define SHARK_OBJECTIVEFUNCTIONS_NEGATIVE_LOG_LIKELIHOOD_H

#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Threading/Algorithms.h>
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
/// \ingroup objfunctions
class NegativeLogLikelihood : public AbstractObjectiveFunction< RealVector, double >
{
public:
	typedef Data<RealVector> DatasetType;

	NegativeLogLikelihood(
		DatasetType const& data,
		AbstractModel<RealVector,RealVector>* model
	):mep_model(model),m_data(data){
		if(mep_model->hasFirstParameterDerivative())
			m_features |= HAS_FIRST_DERIVATIVE;
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NegativeLogLikelihood"; }

	SearchPointType proposeStartingPoint() const{
		return mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	ResultType eval(RealVector const& input) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		m_evaluationCounter++;
		mep_model->setParameterVector(input);
		
		auto map = [this](RealMatrix const& batch){
			double minProb = 1e-100;//numerical stability is only guaranteed for lower bounded probabilities
			RealMatrix predictions = (*mep_model)(batch);
			SIZE_CHECK(predictions.size2() == 1);
			return sum(log(max(predictions,minProb)));
		};
		double error = threading::mapAccumulate( m_data.batches(), 0.0, map, threading::globalThreadPool());
		return -error / m_data.numberOfElements();
	}
	ResultType evalDerivative( 
		SearchPointType const& input, 
		FirstOrderDerivative & derivative 
	) const{
		SIZE_CHECK(input.size() == numberOfVariables());
		m_evaluationCounter++;
		mep_model->setParameterVector(input);
		

		typedef std::pair<double,RealVector> result_type;
		auto map = [this](RealMatrix const& batch){
			double minProb = 1e-100;//numerical stability is only guaranteed for lower bounded probabilities
			RealMatrix predictions;
			boost::shared_ptr<State> state = mep_model->createState();
			mep_model->eval(batch,predictions,*state);
			SIZE_CHECK(predictions.size2() == 1);
			double error = sum(log(max(predictions,minProb)));
			
			//compute coefficients for weighted derivative, handle numerical instabilities
			RealMatrix coeffs(predictions.size1(),predictions.size2(),0.0);
			for(std::size_t j = 0; j != predictions.size1(); ++j){
				for(std::size_t k = 0; k != predictions.size2(); ++k){
					if(predictions(j,k) >= minProb){
						coeffs(j,k) = 1.0/predictions(j,k);
					}
				}
			}
			
			//comptue weighted derivative
			RealVector batchDerivative;
			mep_model->weightedParameterDerivative(
				batch,predictions, coeffs,*state,batchDerivative
			);
			
			//return results
			return result_type(error, std::move(batchDerivative));
		};
		
		//accumulate on the target variables
		double error = 0;
		derivative.resize(input.size());
		derivative.clear();
		auto apply =[&](result_type const& result){
			error += result.first;
			derivative += result.second;
		};
		threading::mapApply( m_data.batches(), map, apply, threading::globalThreadPool());

		std::size_t numElements = m_data.numberOfElements();
		error /= numElements;
		derivative /= numElements;
		derivative *= -1;
		return -error;//negative log likelihood
	}

private:
	AbstractModel<RealVector,RealVector>* mep_model;
	Data<RealVector> m_data;
};

}
#endif
