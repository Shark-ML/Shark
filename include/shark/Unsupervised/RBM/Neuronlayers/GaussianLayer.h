/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_NEURONLAYERS_GAUSSIANLAYER_H
#define SHARK_UNSUPERVISED_RBM_NEURONLAYERS_GAUSSIANLAYER_H

#include <shark/LinAlg/Base.h>
#include <shark/Unsupervised/RBM/StateSpaces/RealSpace.h>
#include <shark/Rng/Normal.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/Math.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h>
#include <shark/Core/OpenMP.h>
namespace shark{

///\brief A layer of Gaussian neurons.
///
/// For a Gaussian neuron/variable the conditional probability distribution of the
/// state of the variable given the state of the other layer is given by a Gaussian
/// distribution with the input of the neuron as mean and unit variance.
class GaussianLayer : public ISerializable, public IParameterizable{
private:
	RealVector m_bias; ///the bias terms associated with the neurons 
public:
	///the state space of this neuron is binary
	typedef RealSpace StateSpace;

	///\brief The sufficient statistics for the Guassian Layer stores the mean of the neuron and the inverse temperature
	typedef RealVector SufficientStatistics;
	///\brief Sufficient statistics of a batch of data.
	typedef Batch<SufficientStatistics>::type StatisticsBatch;
	
	/// \brief Returns the bias values of the units.
	const RealVector& bias()const{
		return m_bias;
	}
	/// \brief Returns the bias values of the units.
	RealVector& bias(){
		return m_bias;
	}
		
	///\brief Resizes this neuron layer.
	///
	///@param newSize number of neurons in the layer
	void resize(std::size_t newSize){
		m_bias.resize(newSize);
	}
	
	///\brief Returns the number of neurons of this layer.
	std::size_t size()const{
		return m_bias.size();
	}
	
	/// \brief Takes the input of the neuron and estimates the expectation of the response of the neuron.
	///
 	/// @param input the batch of inputs of the neuron
	/// @param statistics sufficient statistics containing the mean of the resulting Gaussian distribution
	/// @param beta the inverse Temperature of the RBM (typically 1) for the whole batch
	template<class Input, class BetaVector>
	void sufficientStatistics(Input const& input, StatisticsBatch& statistics,BetaVector const& beta)const{ // \todo: auch hier noch mal namen ueberdenken
		SIZE_CHECK(input.size2() == size());
		SIZE_CHECK(statistics.size2() == size());
		SIZE_CHECK(input.size1() == statistics.size1());
		
		for(std::size_t i = 0; i != input.size1(); ++i){
			noalias(row(statistics,i)) = row(input,i)*beta(i)+m_bias;
		}
	}


	/// \brief Given a the precomputed statistics (the mean of the Gaussian), the elements of the vector are sampled.
	/// This happens either with Gibbs-Sampling or Flip-the-State sampling.
	/// For alpha= 0 gibbs sampling is performed. That is the next state for neuron i is directly taken from the conditional distribution of the i-th neuron. 
	/// In the case of alpha=1, flip-the-state sampling is performed, which takes the last state into account and tries to do deterministically jump 
	/// into states with higher probability. THIS IS NOT IMPLEMENTED YET and alpha is ignored!
	///
	///
	/// @param statistics sufficient statistics containing the mean of the conditional Gaussian distribution of the neurons
	/// @param state the state matrix that will hold the sampled states
	/// @param alpha factor changing from gibbs to flip-the state sampling. 0<=alpha<=1
	/// @param rng the random number generator used for sampling
	template<class Matrix, class Rng>
	void sample(StatisticsBatch const& statistics, Matrix& state, double alpha, Rng& rng) const{
		SIZE_CHECK(statistics.size2() == size());
		SIZE_CHECK(statistics.size1() == state.size1());
		SIZE_CHECK(statistics.size2() == state.size2());
		
		SHARK_CRITICAL_REGION{
			for(std::size_t i = 0; i != state.size1();++i){
				for(std::size_t j = 0; j != state.size2();++j){
					Normal<Rng> normal(rng,statistics(i,j),1.0);
					state(i,j) = normal();
				}
			}
		}
		(void) alpha;
	}
	
	/// \brief Computes the log of the probability of the given states in the conditional distribution
	///
	/// Currently it is only possible to compute the case with alpha=0
	///
	/// @param statistics the statistics of the conditional distribution
	/// @param state the state to check
	template<class Matrix>
	RealVector logProbability(StatisticsBatch const& statistics, Matrix const& state) const{
		SIZE_CHECK(statistics.size2() == size());
		SIZE_CHECK(statistics.size1() == state.size1());
		SIZE_CHECK(statistics.size2() == state.size2());
		
		RealVector logProbabilities(state.size1(),1.0);
		for(std::size_t s = 0; s != state.size1();++s){
			for(std::size_t i = 0; i != state.size2();++i){
				logProbabilities(s) -= 0.5*sqr(statistics(s,i)-state(s,i));
			}
		}
		return logProbabilities;
	}

	/// \brief Transforms the current state of the neurons for the multiplication with the weight matrix of the RBM,
	/// i.e. calculates the value of the phi-function used in the interaction term.
	/// In the case of Gaussian neurons the phi-function is just the identity.
	///
	/// @param state the state matrix of the neuron layer
	/// @return the value of the phi-function
	template<class Matrix>
	Matrix const& phi(Matrix const& state)const{
		SIZE_CHECK(state.size2() == size());
		return state;	
	}

	
	/// \brief Returns the expectation of the phi-function. 
	/// @param statistics the sufficient statistics (the mean of the distribution).
	RealMatrix const& expectedPhiValue(StatisticsBatch const& statistics)const{ 
		SIZE_CHECK(statistics.size2() == size());
		return statistics;	
	}
	/// \brief Returns the mean given the state of the connected layer, i.e. in this case the mean of the Gaussian
	/// 
	/// @param statistics the sufficient statistics of the layer for a whole batch
	RealMatrix const& mean(StatisticsBatch const& statistics)const{ 
		SIZE_CHECK(statistics.size2() == size());
		return statistics;
	}

	/// \brief The energy term this neuron adds to the energy function for a batch of inputs.
	///
	/// @param state the state of the neuron layer
	/// @param beta the inverse temperature of the i-th state
	/// @return the energy term of the neuron layer
	template<class Matrix, class BetaVector>
	RealVector energyTerm(Matrix const& state, BetaVector const& beta)const{
		SIZE_CHECK(state.size2() == size());
		SIZE_CHECK(state.size1() == beta.size());
		//the following code does for batches the equivalent thing to:
		//return beta * inner_prod(m_bias,state) - norm_sqr(state)/2.0;
		
		std::size_t batchSize = state.size1();
		RealVector energies = prod(state,m_bias);
		noalias(energies) *= beta;
		for(std::size_t i = 0; i != batchSize; ++i){
			energies(i) -= norm_sqr(row(state,i))/2.0;
		}
		return energies;
		
	}
	

	///\brief Sums over all possible values of the terms of the energy function which depend on the this layer and returns the logarithmic result.
	///
	///This function is called by Energy when the unnormalized marginal probability of the connected layer is to be computed. 
	///This function calculates the part which depends on the neurons which are to be marginalized out.
	///(In the case of the binary hidden neuron, this is the term \f$  \log \sum_h e^{\vec h^T W \vec v+ \vec h^T \vec c} \f$). 
	///The rest is calculated by the energy function.
	///In the general case of a hidden layer, this function calculates \f$ \log \int_h e^(\phi_h(\vec h)^T W \phi_v(\vec v)+f_h(\vec h) ) \f$ 
	///where f_h  is the energy term of this.
	///
	/// @param inputs the inputs of the neurons they get from the other layer
	/// @param beta the inverse temperature of the RBM
	/// @return the marginal distribution of the connected layer
	template<class Input>
	double logMarginalize(const Input& inputs, double beta) const{
		SIZE_CHECK(inputs.size() == size());
		double lnResult = 0;
		double logNormalizationTerm = std::log(SQRT_2_PI)  - 0.5 * std::log(beta);
		
		for(std::size_t i = 0; i != size(); ++i){
			lnResult += 0.5 * sqr(inputs(i)+m_bias(i))*beta;
			lnResult += logNormalizationTerm;
		}
		return lnResult;
	}


	///\brief Calculates the expectation of the derivatives of the energy term of this neuron layer with respect to it's parameters - the bias weights.
	/// The expectation is taken with respect to the conditional probability distribution of the layer given the state of the connected layer.
	///
	///This function takes a batch of samples and extracts the required informations out of it.
	///@param derivative the derivative with respect to the parameters, the result is added on top of it to accumulate derivatives
	///@param samples the samples from which the informations can be extracted
	template<class Vector, class SampleBatch>
	void expectedParameterDerivative(Vector& derivative, SampleBatch const& samples )const{
		SIZE_CHECK(derivative.size() == size());
		sum_rows(samples.statistics,derivative);
	}

	template<class Vector, class SampleBatch, class Vector2 >
	void expectedParameterDerivative(Vector& derivative, SampleBatch const& samples, Vector2 const& weights )const{
		SIZE_CHECK(derivative.size() == size());
		noalias(derivative) += prod(weights,samples.statistics);
	}
	
	///\brief Calculates the derivatives of the energy term of this neuron layer with respect to it's parameters - the bias weights. 
	///
	///This function takes a batch of samples and extracts the required informations out of it.
	///@param derivative the derivative with respect to the parameters, the result is added on top of it to accumulate derivatives
	///@param samples the sample from which the informations can be extracted
	template<class Vector, class SampleBatch>
	void parameterDerivative(Vector& derivative, SampleBatch const& samples)const{
		SIZE_CHECK(derivative.size() == size());
		sum_rows(samples.state,derivative);
	}
	
	///\brief Calculates the derivatives of the energy term of this neuron layer with respect to it's parameters - the bias weights. 
	///
	///This function takes a batch of samples and calculates a weighted derivative
	///@param derivative the derivative with respect to the parameters, the result is added on top of it to accumulate derivatives
	///@param samples the sample from which the informations can be extracted
	///@param weights the weights for the single sample derivatives
	template<class Vector, class SampleBatch, class WeightVector>
	void parameterDerivative(Vector& derivative, SampleBatch const& samples, WeightVector const& weights)const{
		SIZE_CHECK(derivative.size() == size());
		noalias(derivative) += prod(weights,samples.state);
	}
	
	///\brief Returns the vector with the parameters associated with the neurons in the layer.
	RealVector parameterVector()const{
		return m_bias;
	}

	///\brief Returns the vector with the parameters associated with the neurons in the layer.
	void setParameterVector(RealVector const& newParameters){
		m_bias = newParameters;
	}

	///\brief Returns the number of the parameters associated with the neurons in the layer.
	std::size_t numberOfParameters()const{
		return size();
	}
	
	/// \brief Reads the bias parameters from an archive.
	void read( InArchive & archive ){
		archive >> m_bias;
	}
	/// \brief Writes the bias parameters to an archive.
	void write( OutArchive & archive ) const{
		archive << m_bias;
	}
};

}
#endif
