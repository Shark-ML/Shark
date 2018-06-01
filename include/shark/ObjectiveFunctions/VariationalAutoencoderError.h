/*!
 * 
 *
 * \brief       Variational-autoencoder error function
 * 
 * 
 *
 * \author      O.Krause
 * \date        2017
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
#include <shark/Core/Random.h>

namespace shark{

/// \brief Computes the variational autoencoder error function
///
/// We want to optimize a model \f$ p(x) = \int p(x|z) p(z) dz \f$ where we choose p(z) as a multivariate normal distribution
/// and p(x|z) is an arbitrary model, e.g. a deep neural entwork. The naive solution is sampling from p(z) and then compute the sample
/// average. This will fail when p(z|x) is a very localized distribution and we might need many samples from p(z) to find a sample which is likely under
/// p(z|x). p(z|x) is assumed to be intractable to compute, so we introduce a second model q(z|x), modeling p(z|x) and we want to train
/// it such that it learns the unknown p(z|x). For this a variational lower bound on the likelihood is used and we maximize
/// \f[  log p(x) \leq E_{q(z|x)}[\log p(x|z)] - KL[q(z|x) || p(z)] \f]
/// The first term explains the meaning of variational autoencoder: we first sample z given x using the encoder model q and then decode
/// z to obtain an estimate for x. The only difference to normal autoencoders is that we now have a probabilistic z. The second term ensures that
/// q is learning p(z|x), assuming that we have enough modeling capacity to actually learn it. 
/// See https://arxiv.org/abs/1606.05908 for more background.
///
/// Implementation notice: we assume q(z|x) to be a set of independent gaussian distributions parameterized as
/// \f$ q(z| mu(x), \log \sigma^2(x)) \f$.
/// The provided encoder model q must therefore have twice as many outputs as the decvoder has inputs as
/// the second half of outputs is interpreted as the log of the variance. So if z should be a 100 dimensional variable, q must have 200
/// outputs. The outputs and loss function used for the encoder p is arbitrary, but a SquaredLoss will work well, however also other losses 
/// like pixel probabilities can be used.
/// \ingroup objfunctions

template<class SearchPointType>
class VariationalAutoencoderError : public AbstractObjectiveFunction<SearchPointType, double>
{
private:
	typedef typename SearchPointType::device_type device_type;
	typedef typename SearchPointType::value_type value_type;
	typedef blas::matrix<value_type, blas::row_major, device_type> MatrixType;
public:
	typedef UnlabeledData<SearchPointType> DatasetType;
	typedef AbstractModel<SearchPointType,SearchPointType, SearchPointType> ModelType;

	VariationalAutoencoderError(
		DatasetType const& data,
		ModelType* encoder,
		ModelType* decoder,
		AbstractLoss<SearchPointType, SearchPointType>* visible_loss,
		double lambda = 1.0
	):mep_decoder(decoder), mep_encoder(encoder), mep_loss(visible_loss), m_data(data), m_lambda(lambda){
		if(mep_decoder->hasFirstParameterDerivative() && mep_encoder->hasFirstParameterDerivative())
			this->m_features |= this->HAS_FIRST_DERIVATIVE;
		this->m_features |= this->CAN_PROPOSE_STARTING_POINT;
		this->m_features |= this->IS_NOISY;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "VariationalAutoencoderError"; }

	SearchPointType proposeStartingPoint() const{
		return mep_decoder->parameterVector() | mep_encoder->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_decoder->numberOfParameters() + mep_encoder->numberOfParameters();
	}
	
	MatrixType sampleZ(SearchPointType const& parameters, MatrixType const& batch) const{
		mep_decoder->setParameterVector(subrange(parameters,0,mep_decoder->numberOfParameters()));
		mep_encoder->setParameterVector(subrange(parameters,mep_decoder->numberOfParameters(), numberOfVariables()));
		
		MatrixType hiddenResponse = (*mep_encoder)(batch);
		auto const& mu = columns(hiddenResponse,0,hiddenResponse.size2()/2);
		auto const& log_var = columns(hiddenResponse,hiddenResponse.size2()/2, hiddenResponse.size2());
		//sample random point from distribution
		MatrixType epsilon = blas::normal(*this->mep_rng,mu.size1(), mu.size2(), value_type(0.0), value_type(1.0), device_type());
		return mu + exp(0.5*log_var) * epsilon;
	}

	double eval(SearchPointType const& parameters) const{
		SIZE_CHECK(parameters.size() == numberOfVariables());
		this->m_evaluationCounter++;
		mep_decoder->setParameterVector(subrange(parameters,0,mep_decoder->numberOfParameters()));
		mep_encoder->setParameterVector(subrange(parameters,mep_decoder->numberOfParameters(), numberOfVariables()));
		
		auto const& batch = m_data.batch(random::discrete(*this->mep_rng, std::size_t(0), m_data.numberOfBatches() -1));
		MatrixType hiddenResponse = (*mep_encoder)(batch);
		auto const& mu = columns(hiddenResponse,0,hiddenResponse.size2()/2);
		auto const& log_var = columns(hiddenResponse,hiddenResponse.size2()/2, hiddenResponse.size2());
		//compute kulback leibler divergence term
		double klError = 0.5 * (sum(exp(log_var)) + sum(sqr(mu))  - mu.size1() * mu.size2()  - sum(log_var));
		//sample random point from distribution
		MatrixType epsilon = blas::normal(*this->mep_rng,mu.size1(), mu.size2(), value_type(0.0), value_type(1.0), device_type());
		MatrixType z = mu + exp(0.5*log_var) * epsilon;
		//reconstruct and compute reconstruction error
		MatrixType reconstruction = (*mep_decoder)(z);
		return (m_lambda * (*mep_loss)(batch, reconstruction) + klError) / batch.size1();
	}
	
	
	double evalDerivative( 
		SearchPointType const& parameters, 
		SearchPointType & derivative 
	) const{
		SIZE_CHECK(parameters.size() == numberOfVariables());
		this->m_evaluationCounter++;
		mep_decoder->setParameterVector(subrange(parameters,0,mep_decoder->numberOfParameters()));
		mep_encoder->setParameterVector(subrange(parameters,mep_decoder->numberOfParameters(), numberOfVariables()));
		
		boost::shared_ptr<State> stateEncoder = mep_encoder->createState();
		boost::shared_ptr<State> stateDecoder = mep_decoder->createState();
		auto const& batch = m_data.batch(random::discrete(*this->mep_rng, std::size_t(0), m_data.numberOfBatches() -1));
		MatrixType hiddenResponse;
		mep_encoder->eval(batch,hiddenResponse,*stateEncoder);
		auto const& mu = columns(hiddenResponse,0,hiddenResponse.size2()/2);
		auto const& log_var = columns(hiddenResponse,hiddenResponse.size2()/2, hiddenResponse.size2());
		//compute kulback leibler divergence term
		double klError = 0.5 * (sum(exp(log_var)) + sum(sqr(mu))  - mu.size1() * mu.size2() - sum(log_var));
		MatrixType klDerivative = mu | (0.5 * exp(log_var) - 0.5);
		MatrixType epsilon = blas::normal(*this->mep_rng,mu.size1(), mu.size2(), value_type(0.0), value_type(1.0), device_type());
		MatrixType z = mu + exp(0.5*log_var) * epsilon;
		MatrixType reconstructions;
		mep_decoder->eval(z,reconstructions, *stateDecoder);
		
		
		//compute loss derivative
		MatrixType lossDerivative;
		double recError = m_lambda * mep_loss->evalDerivative(batch,reconstructions,lossDerivative);
		lossDerivative *= m_lambda;
		//backpropagate error from the reconstruction loss to the Decoder
		SearchPointType derivativeDecoder;
		MatrixType backpropDecoder;
		mep_decoder->weightedDerivatives(z,reconstructions, lossDerivative,*stateDecoder, derivativeDecoder, backpropDecoder);
		
		//compute coefficients of the backprop from mep_decoder and the KL-term
		MatrixType backprop=(backpropDecoder | (backpropDecoder * 0.5*(z - mu))) + klDerivative;
		SearchPointType derivativeEncoder;
		mep_encoder->weightedParameterDerivative(batch,hiddenResponse, backprop,*stateEncoder, derivativeEncoder);
	
		derivative.resize(numberOfVariables());
		noalias(derivative) = derivativeDecoder|derivativeEncoder;
		derivative /= batch.size1();
		return (recError + klError) / batch.size1();
	}

private:
	ModelType* mep_decoder;
	ModelType* mep_encoder;
	AbstractLoss<SearchPointType, SearchPointType>* mep_loss;
	UnlabeledData<SearchPointType> m_data;
	double m_lambda;
};

}
#endif
