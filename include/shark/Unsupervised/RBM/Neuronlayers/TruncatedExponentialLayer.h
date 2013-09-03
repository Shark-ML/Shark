/*
*  \par Copyright (c) 1998-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR>
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_NEURONLAYERS_TRUNCATEDEXPONENTIALLAYER_H
#define SHARK_UNSUPERVISED_RBM_NEURONLAYERS_TRUNCATEDEXPONENTIALLAYER_H

#include <shark/Core/ISerializable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/LinAlg/Base.h>
#include <shark/Data/BatchInterfaceAdaptStruct.h>
#include <shark/Unsupervised/RBM/StateSpaces/RealSpace.h>
#include <shark/Unsupervised/RBM/Tags.h>
#include <shark/Rng/TruncatedExponential.h>

namespace shark{
namespace detail{
template<class VectorType>
struct TruncatedExponentialSufficientStatistics{
	VectorType lambda;
	VectorType expMinusLambda;
		
	TruncatedExponentialSufficientStatistics(std::size_t numberOfNeurons)
	:lambda(numberOfNeurons), expMinusLambda(numberOfNeurons){}
	TruncatedExponentialSufficientStatistics(){}
};
}
//auto generate the batch interface for the BinarySufficientStatistics
template<class VectorType>
struct Batch< detail::TruncatedExponentialSufficientStatistics<VectorType> >{
	SHARK_CREATE_BATCH_INTERFACE( 
		detail::TruncatedExponentialSufficientStatistics<VectorType>,
		(VectorType, lambda)(VectorType, expMinusLambda)
	)
};

///\brief A layer of truncated exponential neurons.
///
/// Truncated exponential distributions arise, when the state space of the binary neurons is extended to the
/// real numbers in [0,1]. The conditional distribution of the state of this neurons given the states of the
/// connecred layer is an exponential distribution restricted to [0,1]. 
class TruncatedExponentialLayer : public ISerializable, public IParameterizable{
private:
	RealVector m_bias;
public:
	///the state space of this neuron is real valued, so it can't be enumerated
	typedef RealSpace StateSpace;

	///\brief Stores lambda, the defining parameter of the statistics and also exp(-lambda) since it is used regularly.
	typedef detail::TruncatedExponentialSufficientStatistics<RealVector> SufficientStatistics;
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
	
	/// \brief Takes the input of the neuron and calculates the statistics required to sample from the conditional distribution
	///
 	/// @param input the batch of inputs of the neuron
	/// @param statistics sufficient statistics containing the probabilities of the neurons to be one
	/// @param beta the inverse Temperature of the RBM (typically 1) for the whole batch
	template<class Input, class BetaVector>
	void sufficientStatistics(Input const& input, StatisticsBatch& statistics,BetaVector const& beta)const{ // \todo: auch hier noch mal namen ueberdenken
		SIZE_CHECK(input.size2() == size());
		SIZE_CHECK(statistics.lambda.size2() == size());
		SIZE_CHECK(input.size1() == statistics.lambda.size1());
		
		for(std::size_t i = 0; i != input.size1(); ++i){
			noalias(row(statistics.lambda,i)) = -(row(input,i)+m_bias)*beta(i);
		}
		noalias(statistics.expMinusLambda) = exp(-statistics.lambda);
	}


	///\brief Given the sufficient statistics,(i.e. the parameter lambda and the value of exp(-lambda))
	///  of the neuron, this function samples from the truncated exponential distribution.
	///
	///The truncated exponential function is defined as 
	///\f[ p(x) = \lambda \frac{e^{-lambdax}}{1 - e^{-\lambda}}\f]
	///with x being in the range of [0,1]
	///
	/// @param statistics sufficient statistics for the batch to be computed
	/// @param state the state matrix that will hold the sampled states
	/// @param rng the random number generator used for sampling
	template<class Matrix, class Rng>
	void sample(StatisticsBatch const& statistics, Matrix& state, Rng& rng) const{
		SIZE_CHECK(statistics.lambda.size2() == size());
		SIZE_CHECK(statistics.lambda.size1() == state.size1());
		SIZE_CHECK(statistics.lambda.size2() == state.size2());
		
		for(std::size_t i = 0; i != state.size1();++i){
			for(std::size_t j = 0; j != state.size2();++j){
				double integral = 1.0 - statistics.expMinusLambda(i,j);
				TruncatedExponential<Rng> truncExp(integral,rng,statistics.lambda(i,j));
				state(i,j) = truncExp();
			}
		}
	}

	/// \brief Transforms the current state of the neurons for the multiplication with the weight matrix of the RBM,
	/// i.e. calculates the value of the phi-function used in the interaction term.
	///
	/// @param state the state matrix of the neuron layer
	/// @return the value of the phi-function
	template<class Matrix>
	Matrix const& phi(Matrix const& state)const{
		SIZE_CHECK(state.size2() == size());
		return state;	
	}


	
	/// \brief Returns the conditional expectation of the phi-function given the state of the connected layer.
	/// 
	/// @param statistics the sufficient statistics of the layer
	RealMatrix expectedPhiValue(StatisticsBatch const& statistics)const{ 
		SIZE_CHECK(statistics.lambda.size2() == size());
		SIZE_CHECK(statistics.expMinusLambda.size2() == size());
		std::size_t batchSize=statistics.lambda.size1();
		RealMatrix mean(batchSize,size());
		
		for(std::size_t i = 0; i != batchSize;++i){
			for(std::size_t j = 0; j != size();++j){
				double expML=statistics.expMinusLambda(i,j);
				mean(i,j) = 1.0/statistics.lambda(i,j)-expML/(1.0 - expML);
			}
		}
		return mean;	
	}
	/// \brief Returns the mean of the conditional distribution.
	/// @param statistics the sufficient statistics defining the  conditional distribution
	RealMatrix mean(StatisticsBatch const& statistics)const{ 
		return expectedPhiValue(statistics);
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
		fast_prod(state,m_bias,energies);
		return energies;
	}
	
	///\brief Integrates over the terms of the Energy function which depend on the state of this layer and returns the logarithm of the result.
	///
	///This function is called by Energy when the unnormalized marginal probability of the connected layer is to be computed. 
	///It calculates the part which depends on the neurons which are to be marinalized out.
	///(In the case of the exponential hidden neuron, this is the term \f$ \log \int_h e^{\vec h^T W \vec v+ \vec h^T \vec c} \f$). 
	///
	/// @param inputs the inputs of the neurons they get from the other layer
	/// @param beta the inverse temperature of the RBM
	/// @return the marginal distribution of the connected layer
	template<class Input>
	double logMarginalize(const Input& inputs,double beta) const{
		SIZE_CHECK(inputs.size() == size());
		double lnResult = 0;
		for(std::size_t i = 0; i != size(); ++i){
			double a = (inputs(i)+m_bias(i))*beta;
			//calculates log( (exp(a)-1)/a ). the argument of the log is always positive since for a > 0 is exp(a) > 1 and for a < 0 is exp(a)<1
			//for a = 0 the result is 1 and log(1) = 0
			//so we calculate log( (exp(a)-1)/a ) = log|exp(a)-1| -log|a|
			//we use for a > 0 log|exp(a)-1|=log(exp(a)-1)=a+log(1-exp(-a)) which is numerically more stable if a is big
			// for a< 0, log|exp(a)-1|=log(1-exp(a)) is fine.
			if( a > 1.e-50)
				lnResult += a+std::log(1.0 - std::exp(-a));
			else if( a < 1.e-50)
				lnResult += std::log( 1.0 - std::exp(a));
			lnResult -= std::log(std::abs(a));
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
		sumRows(samples.statistics.probability,derivative);
	}


	///\brief Calculates the derivatives of the energy term of this neuron layer with respect to it's parameters - the bias weights. 
	///
	///This function takes a batch of samples and extracts the required informations out of it.
	///@param derivative the derivative with respect to the parameters, the result is added on top of it to accumulate derivatives
	///@param samples the sample from which the informations can be extracted
	template<class Vector, class SampleBatch>
	void parameterDerivative(Vector& derivative, SampleBatch const& samples)const{
		SIZE_CHECK(derivative.size() == size());
		sumRows(samples.state,derivative);
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
	
	/// \brief Returns the vector with the parameters associated with the neurons in the layer.
	RealVector parameterVector()const{
		return m_bias;
	}

	/// \brief Returns the vector with the parameters associated with the neurons in the layer.
	void setParameterVector(RealVector const& newParameters){
		m_bias = newParameters;
	}

	/// \brief Returns the number of the parameters associated with the neurons in the layer.
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
