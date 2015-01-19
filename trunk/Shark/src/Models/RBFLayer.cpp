/*!
 * 
 *
 * \brief       Implementation of the RBFLayer
 * 
 * 
 *
 * \author      O. Krause
 * \date        2014
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

#include <shark/Models/RBFLayer.h>

using namespace shark;

RBFLayer::RBFLayer()
: m_trainCenters(true), m_trainWidth(true){
	m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
}

RBFLayer::RBFLayer(std::size_t numInput, std::size_t numOutput)
: m_trainCenters(true), m_trainWidth(true){
	setStructure(numInput,numOutput);
	m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
}


void RBFLayer::configure( const PropertyTree & node ){
	std::size_t inputNeurons = node.get<std::size_t>("inputs");
	std::size_t outputNeurons = node.get<std::size_t>("outputs");
	
	bool trainCenters = node.get("trainCenters",true);
	bool trainWidth = node.get("trainWidth",true);
	
	setStructure(inputNeurons,outputNeurons);
	setTrainingParameters(trainCenters,trainWidth);
}
void RBFLayer::setStructure( std::size_t numInput, std::size_t numOutput ){
	m_centers.resize(numOutput,numInput);
	m_gamma.resize(numOutput);
}

RealVector RBFLayer::parameterVector()const{
	RealVector parameters(numberOfParameters());
	if(m_trainCenters && m_trainWidth)
		init(parameters) << toVector(m_centers), log(m_gamma);
	else if( m_trainCenters)
		init(parameters) << toVector(m_centers);
	else if(m_trainWidth)
		parameters = log(m_gamma);
	return parameters;
}


void RBFLayer::setParameterVector(RealVector const& newParameters){
	SIZE_CHECK(newParameters.size() == numberOfParameters());
	
	if(m_trainCenters && m_trainWidth){
		RealVector logGamma(outputSize());
		init(newParameters) >> toVector(m_centers), logGamma;
		setGamma(exp(logGamma));
	}else if( m_trainCenters)
		init(newParameters) >> toVector(m_centers);
	else if(m_trainWidth){
		setGamma(exp(newParameters));
	}
}


std::size_t RBFLayer::numberOfParameters()const{
	std::size_t numParameters=0;
	if(m_trainCenters)
		numParameters += inputSize()*outputSize();
	if(m_trainWidth)
		numParameters += outputSize();
	return numParameters;

}

void RBFLayer::setGamma(RealVector const& gamma){
	SIZE_CHECK(gamma.size() == outputSize());
	m_gamma = gamma;
		
	double logPi = std::log(boost::math::constants::pi<double>());
	m_logNormalization=  inputSize()*0.5*(blas::repeat(logPi,outputSize()) - log(gamma));
}


void RBFLayer::eval(BatchInputType const& patterns, BatchOutputType& output, State& state)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	std::size_t numPatterns = size(patterns);
	output.resize(numPatterns, outputSize());
	
	InternalState& s = state.toState<InternalState>();
	s.resize(numPatterns,outputSize());

	//we need to add separate gamma parameters for every evaluation
	noalias(s.norm2) = distanceSqr(patterns,m_centers);
	
	//every center has it's own value of gamma, so we need to multiply the i-th column 
	//of the norm with m_gamma(i) and to normalize it, we have to subtract the normalization
	// constant.
	noalias(s.p) = exp(
		-element_prod(repeat(m_gamma,numPatterns),s.norm2) 
		- repeat(m_logNormalization,numPatterns)
	);
	
	noalias(output) = s.p;
}

void RBFLayer::weightedParameterDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state,  RealVector& gradient
)const{
	SIZE_CHECK(patterns.size1() == coefficients.size1());
	SIZE_CHECK(coefficients.size2() == outputSize());
	gradient.resize(numberOfParameters());
	InternalState const& s = state.toState<InternalState>();

	//compute d_ij = c_ij * p(x_i|j)
	RealMatrix delta = element_prod(coefficients,s.p);
	RealVector deltaSum = sum_rows(delta);
	
	std::size_t currentParameter = 0;
	if(m_trainCenters){
		//compute the input derivative for every center
		//the formula for the derivative of the i-th center dm_i is
		//dm_j = 2*gamma_j*\sum_i c_{ij} p(x_i|j) (x_i - m_j)
		//     = 2*gamma_j*(\sum_i d_ij*x_i - m_j * \sum_i d_ij)
		
		//the first part is a matrix vector multiplication. this can be cast to a matrix-matrix computation
		//for all centers at the same time!
		//the second part is than just a matrix-diagonal multiplication
		
		blas::dense_matrix_adaptor<double> centerDerivative = blas::adapt_matrix(outputSize(),inputSize(),&gradient(currentParameter));
		//compute first part
		axpy_prod(trans(delta),patterns,centerDerivative);
		//compute second part
		for(std::size_t i = 0; i != outputSize(); ++i){
			noalias(row(centerDerivative,i)) -= deltaSum(i)*row(m_centers,i);
		}
		
		//multiply with 2*gamma
		for(std::size_t i = 0; i != outputSize(); ++i){
			row(centerDerivative,i) *= 2*m_gamma(i);
		}
		
		//move forward
		currentParameter+=outputSize()*inputSize();
	}
	if(m_trainWidth){
		//the derivative of gamma is computed as
		//dgamma_j= \sum_i d_ij/2 -gamma_j *\sum_i d_ij*|x_i-m_j|^2=- gamma_j \sum_j d_ij *n_ij
		//with n_ij = norm2 and d_ij as before
		//the gamma stems from the fact, that the parameter pgamma_i are log encoded 
		//and so gamma_i is in fact e^(pgamma_i) which leads to the fact, that 
		//we have to derive with respect to pgamma_i.
		RealVectorRange gammaDerivative = subrange(gradient,currentParameter,gradient.size());
		noalias(gammaDerivative) = sum_rows(-element_prod(delta,s.norm2));
		noalias(gammaDerivative) = element_prod(gammaDerivative,m_gamma);
		noalias(gammaDerivative) += 0.5*inputSize()*deltaSum;
	}
}


void RBFLayer::setTrainingParameters(bool centers, bool width){
	m_trainCenters=centers;
	m_trainWidth=width;
}

void RBFLayer::read( InArchive & archive ){
	archive >> m_centers;
	archive >> m_gamma;
	archive >> m_logNormalization;;
	archive >> m_trainCenters;
	archive >> m_trainWidth;
}



void RBFLayer::write( OutArchive & archive ) const{
	archive << m_centers;
	archive << m_gamma;
	archive << m_logNormalization;;
	archive << m_trainCenters;
	archive << m_trainWidth;
}

