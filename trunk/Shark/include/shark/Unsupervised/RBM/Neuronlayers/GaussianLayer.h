/*!
 * 
 * \file        GaussianLayer.h
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
#ifndef SHARK_UNSUPERVISED_RBM_NEURONLAYERS_GAUSSIANLAYER_H
#define SHARK_UNSUPERVISED_RBM_NEURONLAYERS_GAUSSIANLAYER_H

#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h>
#include <shark/Unsupervised/RBM/StateSpaces/RealSpace.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Rng/Normal.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/Math.h>

namespace shark{
namespace detail{
template<class VectorType>
struct GaussianSufficientStatistics{
	VectorType mean;
	///unfortunately the beta value needs to be stored since the temperature also changes the variance of the distribution
	double beta; 
		
	GaussianSufficientStatistics(std::size_t numberOfNeurons):mean(numberOfNeurons),beta(1.0){}
	GaussianSufficientStatistics(){}
};
}
//auto generate the batch interface for the GuassianSufficientStatistics
template<class VectorType>
struct Batch< detail::GaussianSufficientStatistics<VectorType> >{
	SHARK_CREATE_BATCH_INTERFACE( detail::GaussianSufficientStatistics<VectorType>,(VectorType, mean)(double, beta))
};

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
	typedef detail::GaussianSufficientStatistics<RealVector> SufficientStatistics;
	///\brief Sufficient statistics of a batch of data.
	typedef Batch<SufficientStatistics>::type StatisticsBatch;
	
	///\brief The type of the State of the Layer.
	typedef RealVector State;
	
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
		SIZE_CHECK(statistics.mean.size2() == size());
		SIZE_CHECK(input.size1() == statistics.mean.size1());
		
		for(std::size_t i = 0; i != input.size1(); ++i){
			noalias(row(statistics.mean,i)) = (row(input,i)+m_bias)*beta(i);
		}
		statistics.beta=beta;
	}


	/// \brief Given a the precomputed statistics (the mean of the Gaussian), the elements of the vector are sampled.
	///
	/// @param statistics sufficient statistics containing the mean of the conditional Gaussian distribution of the neurons
	/// @param state the state matrix that will hold the sampled states
	/// @param rng the random number generator used for sampling
	///
	template<class Matrix, class Rng>
	void sample(StatisticsBatch const& statistics, Matrix& state, Rng& rng) const{
		SIZE_CHECK(statistics.mean.size2() == size());
		SIZE_CHECK(statistics.mean.size1() == state.size1());
		SIZE_CHECK(statistics.mean.size2() == state.size2());
		
		for(std::size_t i = 0; i != state.size1();++i){
			for(std::size_t j = 0; j != state.size2();++j){
				Normal<Rng> normal(rng,statistics.mean(i,j),1.0/statistics.beta(i));
				state(i,j) = normal();
			}
		}
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
		SIZE_CHECK(statistics.mean.size2() == size());
		return statistics.mean;	
	}
	/// \brief Returns the mean given the state of the connected layer, i.e. in this case the mean of the Gaussian
	/// 
	/// @param statistics the sufficient statistics of the layer for a whole batch
	RealMatrix const& mean(StatisticsBatch const& statistics)const{ 
		SIZE_CHECK(statistics.mean.size2() == size());
		return statistics.mean;
	}

	/// \brief The energy term this neuron adds to the energy function for a batch of inputs.
	///
	/// @param state the state of the neuron layer
	/// @return the energy term of the neuron layer
	template<class Matrix>
	RealVector energyTerm(Matrix const& state)const{
		SIZE_CHECK(state.size2() == size());
		//the following code does for batches the equivalent thing to:
		//return inner_prod(m_bias,state) - norm_sqr(state)/2.0;
		
		std::size_t batchSize = state.size1();
		RealVector energies(batchSize);
		axpy_prod(state,m_bias,energies);
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
		
		//std::cout<<inputs<<" "<<inputs-m_bias<<std::endl;
		//std::cout<<remainingTerms<<std::endl;
		
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
		sum_rows(samples.statistics.mean,derivative);
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


	/// \brief Returns the flag of requested values by the expected parameter derivative.
	///
	/// The samples must provide valid informations for this values only. In this case it is only the sufficient statistics of the layer
	GradientFlags flagsExpectedGradient()const{
		GradientFlags flags;
		flags |= RequiresStatistics;
		return flags;
	}

	/// \brief Returns the flag of requested values by the parameter derivative.
	///
	/// The samples must provide valid informations for this values only. In this case it is only the current state of the layer
	GradientFlags flagsGradient()const{
		GradientFlags flags;
		flags |= RequiresState;
		return flags;
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
