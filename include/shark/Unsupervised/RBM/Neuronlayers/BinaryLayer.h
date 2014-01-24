/*!
 * 
 * \file        BinaryLayer.h
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_UNSUPERVISED_RBM_NEURONLAYERS_BINARYLAYER_H
#define SHARK_UNSUPERVISED_RBM_NEURONLAYERS_BINARYLAYER_H

#include <shark/Core/ISerializable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h>
#include <shark/Rng/Bernoulli.h>
#include <shark/Unsupervised/RBM/StateSpaces/TwoStateSpace.h>
#include <shark/Unsupervised/RBM/Tags.h>

namespace shark{
namespace detail{
template<class VectorType>
struct BinarySufficientStatistics{
	VectorType probability;
	
	BinarySufficientStatistics(std::size_t numberOfNeurons):probability(numberOfNeurons){}
	BinarySufficientStatistics(){}
};
}
//auto generate the batch interface for the BinarySufficientStatistics
template<class VectorType>
struct Batch< detail::BinarySufficientStatistics<VectorType> >{
	SHARK_CREATE_BATCH_INTERFACE( detail::BinarySufficientStatistics<VectorType>,(VectorType,probability))
};


///\brief Layer of binary units taking values in {0,1}. 

///A neuron in a Binary Layer takes values in {0,1} and the conditional probability to be 1 
///given the states of the neurons in the connected layer is determined by the sigmoid function
///and the input it gets from the connected layer.   
class BinaryLayer : public ISerializable, public IParameterizable{
private:
	///\brief The bias terms associated with the neurons.
	RealVector m_bias; 
public:
	///\brief The state space of this neuron is binary.
	typedef BinarySpace StateSpace;

	///\brief The sufficient statistics for the Binary Layer store the probability for a neuron to be on
	typedef detail::BinarySufficientStatistics<RealVector> SufficientStatistics;
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
	///	For binary neurons the expectation is identical with the conditional probability for the neuron to be on given the state of the connected layer.
	///
 	/// @param input the batch of inputs of the neuron
	/// @param statistics sufficient statistics containing the probabilities of the neurons to be one
	/// @param beta the inverse Temperature of the RBM (typically 1) for the whole batch
	template<class Input, class BetaVector>
	void sufficientStatistics(Input const& input, StatisticsBatch& statistics,BetaVector const& beta)const{ // \todo: auch hier noch mal namen ueberdenken
		SIZE_CHECK(input.size2() == size());
		SIZE_CHECK(statistics.probability.size2() == size());
		SIZE_CHECK(input.size1() == statistics.probability.size1());
		
		for(std::size_t i = 0; i != input.size1(); ++i){
			noalias(row(statistics.probability,i)) = sigmoid((row(input,i)+m_bias)*beta(i));
		}
	}


	/// \brief Given the precomputed probability of the neurons to be in state one, the states of the neurons are sampled.
	///
	/// @param statistics sufficient statistics containing the probabilities of the neurons to be one
	/// @param state the state vector that shell hold the sampled states
	/// @param rng the random number generator used for sampling
	///
	template<class Matrix, class Rng>
	void sample(StatisticsBatch const& statistics, Matrix& state, Rng& rng) const{
		SIZE_CHECK(statistics.probability.size2() == size());
		SIZE_CHECK(statistics.probability.size1() == state.size1());
		SIZE_CHECK(statistics.probability.size2() == state.size2());
		
		Bernoulli<Rng> coinToss(rng,0.5);
		for(std::size_t i = 0; i != state.size1();++i){
			for(std::size_t j = 0; j != state.size2();++j){
				state(i,j) = coinToss(statistics.probability(i,j));
			}
		}
	}

	/// \brief Transforms the current state of the neurons for the multiplication with the weight matrix of the RBM,
	/// i.e. calculates the value of the phi-function used in the interaction term.
	/// In the case of binary neurons the phi-function is just the identity.
	///
	/// @param state the state matrix of the neuron layer
	/// @return the value of the phi-function
	template<class Matrix>
	Matrix const& phi(Matrix const& state)const{
		SIZE_CHECK(state.size2() == size());
		return state;	
	}

	
	/// \brief Returns the conditional expectation of the phi-function given the state of the connected layer,
	/// i.e. in this case the probabilities of the neurons having state one.
	/// 
	/// @param statistics the sufficient statistics of the layer
	RealMatrix const& expectedPhiValue(StatisticsBatch const& statistics)const{ 
		return statistics.probability;	
	}

	/// \brief Returns the mean given the state of the connected layer, i.e. in this case the probabilities of the neurons having state one.
	/// 
	/// @param statistics the sufficient statistics of the layer for a whole batch
	RealMatrix const& mean(StatisticsBatch const& statistics)const{ 
		SIZE_CHECK(statistics.probability.size2() == size());
		return statistics.probability;
	}

	/// \brief Returns the energy term this neuron adds to the energy function.
	///
	/// @param state the state of the neuron layer 
	/// @return the energy term of the neuron layer
	template<class Matrix>
	RealVector energyTerm(Matrix const& state)const{
		SIZE_CHECK(state.size2() == size());
		//the following code does for batches the equivalent thing to:
		//return inner_prod(m_bias,state)
		
		RealVector energies(state.size1());
		axpy_prod(state,m_bias,energies);
		return energies;
	}
	

	///\brief Sums over all possible values of the terms of the energy function which depend on the this layer and returns the logarithmic result.
	///
	///This function is called by Energy when the unnormalized marginal probability of the connected layer is to be computed. 
	///This function calculates the part which depends on the neurons which are to be marginalized out.
	///(In the case of the binary hidden neuron, this is the term \f$ \sum_h e^{\vec h^T W \vec v+ \vec h^T \vec c} \f$). 
	///The rest is calculated by the energy function.
	///In the general case of a hidden layer, this function calculates \f$ \int_h e^(\phi_h(\vec h)^T W \phi_v(\vec v)+f_h(\vec h) ) \f$ 
	///where f_h  is the energy term of this layer.
	///
	/// @param inputs the inputs of the neurons they get from the other layer
	/// @param beta the inverse temperature of the RBM
	/// @return the marginal distribution of the connected layer
	template<class Input>
	double logMarginalize(Input const& inputs, double beta) const{
		SIZE_CHECK(inputs.size() == size());
		long double logFactorization = 0;
		for(std::size_t i = 0; i != inputs.size(); ++i){
			double arg = (inputs(i)+m_bias(i))*beta;
			logFactorization += softPlus(arg);
		}
		return logFactorization;
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
		sum_rows(samples.statistics.probability,derivative);
	}
	
	///\brief Calculates the expectation of the derivatives of the energy term of this neuron layer with respect to it's parameters - the bias weights.
	/// The expectation is taken with respect to the conditional probability distribution of the layer given the state of the connected layer.
	///
	///This function takes a batch of samples and weights the results
	///@param derivative the derivative with respect to the parameters, the result is added on top of it to accumulate derivatives
	///@param samples the samples from which the informations can be extracted
	///@param weights The weights for alle samples
	template<class Vector, class SampleBatch, class WeightVector>
	void expectedParameterDerivative(Vector& derivative, SampleBatch const& samples, WeightVector const& weights )const{
		SIZE_CHECK(derivative.size() == size());
		axpy_prod(trans(samples.statistics.probability),weights,derivative,false);
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
		axpy_prod(trans(samples.state),weights,derivative,false);
	}
	
	/// \brief Returns the flag of requested values by the expected parameter derivative.
	GradientFlags flagsExpectedGradient()const{
		GradientFlags flags;
		flags |= RequiresStatistics;
		return flags;
	}

	/// \brief Returns the flag of requested values by the parameter derivative.
	GradientFlags flagsGradient()const{
		GradientFlags flags;
		flags |= RequiresState;
		return flags;
	}
	
	/// \brief Returns the vector with the parameters associated with the neurons in the layer, i.e. the bias vector.
	RealVector parameterVector()const{
		return m_bias;
	}

	/// \brief Sets the parameters associated with the neurons in the layer, i.e. the bias vector.
	void setParameterVector(RealVector const& newParameters){
		m_bias = newParameters;
	}

	/// \brief Returns the number of the parameters associated with the neurons in the layer.
	std::size_t numberOfParameters()const{
		return size();
	}
	
	/// \brief Reads the bias vector from an archive.
	///
	/// @param archive the archive
	void read( InArchive & archive ){
		archive >> m_bias;
	}

	/// \brief Writes the bias vector to an archive.
	///
	/// @param archive the archive
	void write( OutArchive & archive ) const{
		archive << m_bias;
	}
};

}
#endif
