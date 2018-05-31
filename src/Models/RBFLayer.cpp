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
#define SHARK_COMPILE_DLL
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

void RBFLayer::setStructure( std::size_t numInput, std::size_t numOutput ){
	m_centers.resize(numOutput,numInput);
	m_gamma.resize(numOutput);
}

RealVector RBFLayer::parameterVector()const{
	if(m_trainCenters && m_trainWidth)
		return  to_vector(m_centers) | log(m_gamma);
	else if( m_trainCenters)
		return to_vector(m_centers);
	else if(m_trainWidth)
		return log(m_gamma);
	return RealVector();
}


void RBFLayer::setParameterVector(RealVector const& newParameters){
	SIZE_CHECK(newParameters.size() == numberOfParameters());
	std::size_t pos = 0;
	if( m_trainCenters){
		pos = inputShape().numElements() * outputShape().numElements();
		noalias(to_vector(m_centers)) = subrange(newParameters,0,pos);
	}
	if(m_trainWidth){
		setGamma(exp(subrange(newParameters,pos,newParameters.size())));
	}
}


std::size_t RBFLayer::numberOfParameters()const{
	std::size_t numParameters=0;
	if(m_trainCenters)
		numParameters += inputShape().numElements() * outputShape().numElements();
	if(m_trainWidth)
		numParameters += outputShape().numElements();
	return numParameters;

}

void RBFLayer::setGamma(RealVector const& gamma){
	SIZE_CHECK(gamma.size() == outputShape().numElements());
	m_gamma = gamma;
		
	double logPi = std::log(boost::math::constants::pi<double>());
	m_logNormalization=  inputShape().numElements()*0.5*(logPi - log(gamma));
}

void RBFLayer::eval(BatchInputType const& patterns, BatchOutputType& output, State& state)const{
	SIZE_CHECK(patterns.size2() == inputShape().numElements());
	std::size_t numPatterns = patterns.size1();
	output.resize(numPatterns, outputShape().numElements());
	
	InternalState& s = state.toState<InternalState>();
	s.resize(numPatterns,outputShape().numElements());

	//we need to add separate gamma parameters for every evaluation
	noalias(s.norm2) = distanceSqr(patterns,m_centers);
	
	//every center has it's own value of gamma, so we need to multiply the i-th column 
	//of the norm with m_gamma(i) and to normalize it, we have to subtract the normalization
	// constant.
	noalias(output) = exp(
		-element_prod(repeat(m_gamma,numPatterns),s.norm2) 
		- repeat(m_logNormalization,numPatterns)
	);
}

void RBFLayer::weightedParameterDerivative(
	BatchInputType const& patterns, BatchOutputType const& outputs, 
	BatchOutputType const& coefficients, State const& state,  RealVector& gradient
)const{
	SIZE_CHECK(patterns.size1() == coefficients.size1());
	SIZE_CHECK(coefficients.size2() == outputShape().numElements());
	gradient.resize(numberOfParameters());
	InternalState const& s = state.toState<InternalState>();

	//compute d_ij = c_ij * p(x_i|j)
	RealMatrix delta = coefficients * outputs;
	RealVector deltaSum = sum(as_columns(delta));
	
	std::size_t currentParameter = 0;
	if(m_trainCenters){
		//compute the input derivative for every center
		//the formula for the derivative of the i-th center dm_i is
		//dm_j = 2*gamma_j*\sum_i c_{ij} p(x_i|j) (x_i - m_j)
		//     = 2*gamma_j*(\sum_i d_ij*x_i - m_j * \sum_i d_ij)
		
		//the first part is a matrix vector multiplication. this can be cast to a matrix-matrix computation
		//for all centers at the same time!
		//the second part is than just a matrix-diagonal multiplication
		
		blas::dense_matrix_adaptor<double> centerDerivative = blas::adapt_matrix(m_centers.size1(),m_centers.size2(),&gradient(currentParameter));
		noalias(centerDerivative) = prod(trans(delta),patterns);
		//compute second part
		for(std::size_t i = 0; i != m_centers.size1(); ++i){
			noalias(row(centerDerivative,i)) -= deltaSum(i)*row(m_centers,i);
			row(centerDerivative,i) *= 2*m_gamma(i);
		}
		
		//move forward
		currentParameter += m_centers.size1() * m_centers.size2();
	}
	if(m_trainWidth){
		//the derivative of gamma is computed as
		//dgamma_j= \sum_i d_ij/2 -gamma_j *\sum_i d_ij*|x_i-m_j|^2=- gamma_j \sum_j d_ij *n_ij
		//with n_ij = norm2 and d_ij as before
		//the gamma stems from the fact, that the parameter pgamma_i are log encoded 
		//and so gamma_i is in fact e^(pgamma_i) which leads to the fact, that 
		//we have to derive with respect to pgamma_i.
		auto gammaDerivative = subrange(gradient,currentParameter,gradient.size());
		noalias(gammaDerivative) = sum(as_columns(-delta * s.norm2));
		noalias(gammaDerivative) = element_prod(gammaDerivative,m_gamma);
		noalias(gammaDerivative) += 0.5*m_centers.size2()*deltaSum;
	}
}


void RBFLayer::setTrainingParameters(bool centers, bool width){
	m_trainCenters=centers;
	m_trainWidth=width;
}

void RBFLayer::read( InArchive & archive ){
	archive >> m_centers;
	archive >> m_gamma;
	archive >> m_logNormalization;
	archive >> m_trainCenters;
	archive >> m_trainWidth;
}



void RBFLayer::write( OutArchive & archive ) const{
	archive << m_centers;
	archive << m_gamma;
	archive << m_logNormalization;
	archive << m_trainCenters;
	archive << m_trainWidth;
}

