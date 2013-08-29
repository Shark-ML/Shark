/*!
*  \brief Implementation of the RBFNet
*
*  \author  O. Krause
*  \date    2010
*
*  \par Copyright (c) 1999-2001:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
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
*  
*/

//#include <boost/serialization/vector.hpp>
#include <shark/LinAlg/BLAS/Initialize.h>
#include <shark/LinAlg/BLAS/StorageAdaptors.h>
#include <shark/Models/RBFNet.h>

using namespace shark;

RBFNet::RBFNet():m_inputNeurons(0),m_outputNeurons(0),
m_trainLinear(true),m_trainCenters(true),m_trainWidth(true){
	m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
}

RBFNet::RBFNet(std::size_t numInput, std::size_t numHidden, std::size_t numOutput)
:m_trainLinear(true),m_trainCenters(true),m_trainWidth(true){
	setStructure(numInput,numHidden,numOutput);
	m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
}


void RBFNet::configure( const PropertyTree & node ){
	std::size_t inputNeurons = node.get<std::size_t>("inputs");
	std::size_t hiddenNeurons = node.get<std::size_t>("hidden");
	std::size_t outputNeurons = node.get<std::size_t>("outputs");
	
	bool trainLinear = node.get("trainLinear",true);
	bool trainCenters = node.get("trainCenters",true);
	bool trainWidth = node.get("trainWidth",true);
	
	setStructure(inputNeurons,hiddenNeurons,outputNeurons);
	setTrainingParameters(trainLinear,trainCenters,trainWidth);
}
void RBFNet::setStructure( std::size_t numInput, std::size_t numHidden, std::size_t numOutput ){
	m_inputNeurons = numInput;
	m_outputNeurons = numOutput;
	m_centers.resize(numHidden,numInput);
	m_linearWeights.resize(numOutput,numHidden);
	m_bias.resize(numOutput);
	m_gamma.resize(numHidden);
}

RealVector RBFNet::parameterVector()const{
	RealVector parameters(numberOfParameters());
	std::size_t start = 0;
	std::size_t end = 0;
	if(m_trainLinear){
		end = numHiddens()*m_outputNeurons+m_outputNeurons;
		init(subrange(parameters,start,end)) << toVector(m_linearWeights),m_bias;
		start = end;
	}
	if(m_trainCenters){
		end += m_inputNeurons*numHiddens();
		init(subrange(parameters,start,end)) << toVector(m_centers);
		start=end;
	}
	if(m_trainWidth){
		end = numberOfParameters();
		init(subrange(parameters,start,end)) << log(m_gamma);
	}
	return parameters;
}


void RBFNet::setParameterVector(RealVector const& newParameters){
	SIZE_CHECK(newParameters.size()==numberOfParameters());
	std::size_t start = 0;
	std::size_t end = 0;
	if(m_trainLinear){
		end = numHiddens()*m_outputNeurons+m_outputNeurons;
		init(subrange(newParameters,start,end)) >> toVector(m_linearWeights),m_bias;
		start = end;
	}
	if(m_trainCenters){
		end = start+m_inputNeurons*numHiddens();
		init(subrange(newParameters,start,end)) >> toVector(m_centers);
		start=end;
	}
	if(m_trainWidth)
	{
		end = newParameters.size();
		init(subrange(newParameters,start,end)) >> m_gamma;
		noalias(m_gamma) = exp(m_gamma);
	}
}


std::size_t RBFNet::numberOfParameters()const{
	std::size_t numParameters=0;
	if(m_trainLinear)
		numParameters += numHiddens()*m_outputNeurons+m_outputNeurons;
	if(m_trainCenters)
		numParameters += m_inputNeurons*numHiddens();
	if(m_trainWidth)
		numParameters += numHiddens();

	return numParameters;

}

void RBFNet::computeGaussianResponses(BatchInputType const& patterns, InternalState& state)const{
	std::size_t numPatterns = patterns.size1();
	std::size_t numNeurons = m_gamma.size();
	state.resize(numPatterns,numNeurons);

	//we need to add separate gamma parameters for every evaluation
	noalias(state.norm2) = distanceSqr(patterns,m_centers);
	
	//every center has it's own value of gamma, so we need to multiply the i-th column 
	//of the norm with m_gamma(i)
	noalias(state.expNorm) = exp(-element_prod(repeat(m_gamma,numPatterns),state.norm2));
}


void RBFNet::eval(BatchInputType const& patterns, BatchOutputType& output, State& state)const{
	SIZE_CHECK(patterns.size2() == m_inputNeurons);
	std::size_t numPatterns = size(patterns);
	output.resize(numPatterns, m_outputNeurons);
	InternalState& s = state.toState<InternalState>();
	
	//evaluate kernel expNorm and store them in the intermediates
	computeGaussianResponses(patterns,s);
	//evaluate the linear part of the network
	noalias(output) = repeat(m_bias,numPatterns);
	fast_prod(s.expNorm,trans(m_linearWeights),output,true);
}

void RBFNet::weightedParameterDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state,  RealVector& gradient
)const{
	SIZE_CHECK(patterns.size1() == coefficients.size1());
	SIZE_CHECK(coefficients.size2() == outputSize());
	InternalState const& s = state.toState<InternalState>();
	//this is basically a generalized backprop like in FFNet, only with the difference,
	//that we know that the output is linear and we have no shortcuts. The only theoretical difference between
	//FFNet-Backprop and this is, that here the neurons also have weights
	//and are additionally depending on center values.
	
	std::size_t numPatterns = patterns.size1();
	std::size_t numNeurons = m_gamma.size();

	gradient.resize(numberOfParameters());

	std::size_t currentParameter=0;//current parameter which is evaluated
	
	//first evaluate the derivatives of the linear part if enabled
	if(m_trainLinear){
		//interpret the linear part of the parameter vector as matrix
		blas::FixedDenseMatrixProxy<double> weightDerivative = blas::makeMatrix(m_outputNeurons,numNeurons,&gradient(0));
		fast_prod(trans(coefficients),s.expNorm,weightDerivative);
		currentParameter += m_outputNeurons*numNeurons;
		//bias
		RealVectorRange biasDerivative = subrange(gradient,currentParameter,currentParameter+outputSize());
		noalias(biasDerivative) = sumRows(coefficients);
		currentParameter += outputSize();
	}

	//test whether training of the distributions is necessary
	if(!(m_trainCenters || m_trainWidth))
		return;
		
	//this now is the backpropagation step, see FFNet for more explanations of how this works
	//since we have a very special output layer, the delta values are easy to compute - they are just the linearWeights
	//themselves.So we don't have to compute them at allowed
	
	//We have to calculate the delta-values first from the coefficients
	//calculates delta_j=sum_i c_i*w_{ij}<=>delta=coefficients*linearWeights for every element of the pattern
	RealMatrix delta(numPatterns,numNeurons);
	fast_prod(coefficients,m_linearWeights,delta);
	
	//in the next steps, we will compute two derivates of the exponential fucntion.
	//It has the nice property, that the exponentials of exp(...) also have exp(...) in the
	//result. so we can regard the exp function as an additional layer by itself and just do another step of backprop
	//this saves us the multiplication later on 3 times!
	noalias(delta) = element_prod(delta,s.expNorm);

	if(m_trainCenters){
		//compute the input derivative for every center
		//the formula for the derivative of the i-th center dc_i is
		//dc_i = 2*gamma_i*\sum_j d_ij(x_j - c_i)
		//     = 2*gamma_i*(\sum_j d_ij*x_j - c_i * \sum_j d_ij * w_ij)
		//d_ij=delta x_j=patterns c_i = centers
		//the first part is a matrix vector multiplication. this can be cast to a matrix-matrix computation
		//for all centers at the same time!
		//the second part is than just a matrix-diagonal multiplication
		
		blas::FixedDenseMatrixProxy<double> centerDerivative = blas::makeMatrix(numNeurons,inputSize(),&gradient(currentParameter));
		//compute first part
		fast_prod(trans(delta),patterns,centerDerivative);
		//compute second part
		RealVector weightSum = sumRows(delta);
		for(std::size_t i = 0; i != numNeurons; ++i){
			noalias(row(centerDerivative,i)) -= weightSum(i)*row(m_centers,i);
		}
		
		//multiply with 2*gamma
		for(std::size_t i = 0; i != numNeurons; ++i){
			row(centerDerivative,i) *= 2*m_gamma(i);
		}
		
		//move forward
		currentParameter+=numNeurons*inputSize();
	}
	if(m_trainWidth){
		//the derivative of gamma is computed as
		//dgamma_i= -gamma*\sum_j d_ij*|x_j-c_i|^2=-log(gamma)\sum_j d_ij *n_ij
		//with n_ij = norm2 and d_ij = delta
		//the gamma stems from the fact, that the parameter pgamma_i are log encoded 
		//and so gamma_i is in fact e^(pgamma_i) which leads to the fact, that 
		//we have to derive with respect to pgamma_i.
		RealVectorRange gammaDerivative = subrange(gradient,currentParameter,gradient.size());
		noalias(gammaDerivative) = sumRows(-element_prod(delta,s.norm2));
		noalias(gammaDerivative) = element_prod(gammaDerivative,m_gamma);
	}
}


void RBFNet::setTrainingParameters(bool linear,bool centers, bool width){
	m_trainLinear=linear;
	m_trainCenters=centers;
	m_trainWidth=width;
}

void RBFNet::read( InArchive & archive ){
	archive >> m_inputNeurons;
	archive >> m_outputNeurons;
	archive >> m_centers;
	archive >> m_linearWeights;
	archive >> m_bias;
	archive >> m_gamma;
	archive >> m_trainLinear;
	archive >> m_trainCenters;
	archive >> m_trainWidth;
}



void RBFNet::write( OutArchive & archive ) const{
	archive << m_inputNeurons;
	archive << m_outputNeurons;
	archive << m_centers;
	archive << m_linearWeights;
	archive << m_bias;
	archive << m_gamma;
	archive << m_trainLinear;
	archive << m_trainCenters;
	archive << m_trainWidth;
}

