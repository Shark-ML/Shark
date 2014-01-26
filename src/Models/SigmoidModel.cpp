//===========================================================================
/*!
 * 
 *
 * \brief       SigmoidModel
 * 
 * 
 *
 * \author      O.Krause, T.Glasmachers, M.Tuma
 * \date        2010-2011
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
//===========================================================================
#include <shark/Models/SigmoidModel.h>

#include <boost/math/special_functions/sign.hpp>

#include <cmath>

using namespace shark;
using namespace std;

SigmoidModel::SigmoidModel( bool transform_for_unconstrained )
:m_parameters(2),
 m_useOffset(true),
 m_transformForUnconstrained(transform_for_unconstrained),
 m_minLogValue(-230) {
	m_features |= HAS_FIRST_PARAMETER_DERIVATIVE;
	m_features |= HAS_FIRST_INPUT_DERIVATIVE;
	m_parameters(0) = 1.0; //slope
	m_parameters(1) = 0.0; //offset
}

RealVector SigmoidModel::parameterVector() const {
	RealVector ret(2);
	if ( m_transformForUnconstrained ) {
		ret(0) = std::log(m_parameters(0));
		if ( ret(0) < m_minLogValue ) ret(0) = m_minLogValue;
	} else {
		ret(0) = m_parameters(0);
	}
	ret(1) = m_parameters(1);
	return ret;
}

void SigmoidModel::setParameterVector(RealVector const& newParameters){
	SIZE_CHECK( newParameters.size() == 2 );
	if ( m_transformForUnconstrained ) {
		m_parameters(0) = std::exp( newParameters(0) );
	} else {
		if ( newParameters(0) < 0.0 ) {
			m_parameters(0) = 0.0;
		} else {
			m_parameters(0) = newParameters(0);
		}
	}
	if ( m_useOffset == false ) {
		m_parameters(1) = 0.0;
	} else {
		m_parameters(1) = newParameters(1);
	}
}

void SigmoidModel::setOffsetActivity( bool enable_offset ) {
	m_useOffset = enable_offset;
	if ( m_useOffset == false )
		m_parameters(1) = 0.0;
}

double SigmoidModel::sigmoid(double x)const{
	return 1.0 / (1.0 + std::exp(-x));
}

double SigmoidModel::sigmoidDerivative(double gx)const{
	return gx *(1 - gx);
}

void SigmoidModel::eval(BatchInputType const&patterns, BatchOutputType& outputs)const{
	SIZE_CHECK( patterns.size2() == 1 );
	outputs.resize(patterns.size1(),1);
	//note that because of the way the intermediate result is passed to the sigmoid member function
	// (facilitating derivatives and sub-classes), we here have to substract the bias parameter.
	noalias(column(outputs,0)) = column(patterns,0)*m_parameters(0) - blas::repeat(m_parameters(1),patterns.size1());
	for(std::size_t i = 0; i != patterns.size1(); ++i)
		outputs(i,0) = sigmoid(outputs(i,0));
}

void SigmoidModel::eval(BatchInputType const&patterns, BatchOutputType& outputs, State& state)const{
	eval(patterns,outputs);
	InternalState& s = state.toState<InternalState>();
	s.resize(patterns.size1());
	noalias(s.result) = column(outputs,0);
}

void SigmoidModel::weightedParameterDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state, RealVector& gradient
)const{
	SIZE_CHECK( patterns.size2() == 1 );
	SIZE_CHECK( coefficients.size2() == 1 );
	SIZE_CHECK( coefficients.size1() == patterns.size1() );
	InternalState const& s = state.toState<InternalState>();
	gradient.resize(2);
	gradient(0)=0;
	gradient(1)=0;
	//calculate derivative
	for(std::size_t i = 0; i != patterns.size1(); ++i){
		double derivative = sigmoidDerivative( s.result(i) );
		double slope= coefficients(i,0)*derivative*patterns(i,0); //w.r.t. slope
		if ( m_transformForUnconstrained )
			slope *= m_parameters(0);
		gradient(0)+=slope;
		if ( m_useOffset  ) {
			gradient(1) -= coefficients(i,0)*derivative; //w.r.t. bias parameter
		}
	}
}

void SigmoidModel::weightedInputDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state, BatchInputType& derivatives
)const{
	SIZE_CHECK( patterns.size2() == 1 );
	SIZE_CHECK( coefficients.size2() == 1 );
	SIZE_CHECK( coefficients.size1() == patterns.size1() );
	InternalState const& s = state.toState<InternalState>();
	std::size_t numPatterns= patterns.size1();
	derivatives.resize( numPatterns,1);
	//calculate derivative
	for(std::size_t i = 0; i != numPatterns; ++i){
		double der = sigmoidDerivative( s.result(i) );
		derivatives(i,0) = coefficients(i,0) * der * m_parameters(0);
	}
}

/// From ISerializable, reads a model from an archive
void SigmoidModel::read( InArchive & archive ){
	archive >> m_parameters;
	archive >> m_useOffset;
}

/// From ISerializable, writes a model to an archive
void SigmoidModel::write( OutArchive & archive ) const{
	archive << m_parameters;
	archive << m_useOffset;
}

void SigmoidModel::setMinLogValue( double logvalue ) {
	SHARK_ASSERT( logvalue < 0 );
	m_minLogValue = logvalue;
}


////////////////////////////////////////////////////////////


SimpleSigmoidModel::SimpleSigmoidModel( bool transform_for_unconstrained ) : SigmoidModel( transform_for_unconstrained )
{ }

double SimpleSigmoidModel::sigmoid(double x)const{
	return 0.5 * x / (1.0 + std::abs(x)) + 0.5;
}
double SimpleSigmoidModel::sigmoidDerivative(double gx)const{
	return 0.5*sqr(1 - boost::math::sign(gx) * gx);
}


////////////////////////////////////////////////////////////


TanhSigmoidModel::TanhSigmoidModel( bool transform_for_unconstrained ) : SigmoidModel( transform_for_unconstrained )
{ }

double TanhSigmoidModel::sigmoid(double x)const{
	return 0.5*std::tanh(x)+0.5;
}

double TanhSigmoidModel::sigmoidDerivative(double gx)const{
	return 0.5*(1 - gx * gx);
}
