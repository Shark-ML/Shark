//===========================================================================
/*!
 * 
 * \file        NegativeClassificationLogLikelihood.cpp
 *
 * \brief       negative logarithm of the likelihood of a probabilistic binary classification model
 * 
 * 
 *
 * \author      T. Glasmachers, O.Krause, M. Tuma
 * \date        2011
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

#include <boost/math/special_functions/log1p.hpp>
#include <shark/ObjectiveFunctions/Loss/NegativeClassificationLogLikelihood.h>

using namespace shark;
using namespace std;

NegativeClassificationLogLikelihood::NegativeClassificationLogLikelihood()
{
	this->m_features |= base_type::HAS_FIRST_DERIVATIVE;
	m_minArgToLog = 1e-100;
	m_minLogReturnVal = std::log( m_minArgToLog );
	m_minArgToLog1p = 1e-10;
	m_minLog1pReturnVal = std::log( m_minArgToLog1p );
}
double NegativeClassificationLogLikelihood::evalErrorBinary(unsigned int target, double prediction)const{
	if ( target == 1) {
		if ( prediction < m_minArgToLog ) { //prediction shouldn't be zero (even though it is unlikely to)
			return -m_minLogReturnVal;
		}
		else {
			return -std::log( prediction );
		}
	} else {
		if ( 1 < prediction + m_minArgToLog1p ) { //prediction shouldn't be one (even though it is unlikely to)
			return -m_minLog1pReturnVal;
		}
		else{
			return -boost::math::log1p( -prediction ); //return -log(1-p(x)). using log1p to avoid numerical instabilities
		}
	}
}

double NegativeClassificationLogLikelihood::eval(UIntVector const& target, RealMatrix const& predictions) const {
	SIZE_CHECK(target.size() == predictions.size1());
	std::size_t psize = predictions.size2();
	
	double error = 0;
	if ( psize==1 ) {
		for(std::size_t i = 0; i != predictions.size1();++i){
			RANGE_CHECK(target(i) < 2);
			error += evalErrorBinary(target(i),predictions(i,0));
		}
	} else {
		for(std::size_t i = 0; i != predictions.size1();++i){
			RANGE_CHECK(target(i) < psize);
			double prediction = predictions(i,target(i));
			if ( prediction < m_minArgToLog ) { //prediction shouldn't be zero (even though it is unlikely to)
				error+= -m_minLogReturnVal;
			}
			else{
				error+= -std::log( prediction );
			}
		}
	}
	return error;
}


double NegativeClassificationLogLikelihood::evalDerivative(UIntVector const& target, RealMatrix const& predictions, RealMatrix& gradient) const {
	SIZE_CHECK(target.size() == predictions.size1());
	std::size_t psize = predictions.size2();
	std::size_t batchSize = predictions.size1();
	
	double error = 0;
	gradient.resize(batchSize, psize );
	
	if( psize==1 ){
		for(std::size_t i = 0; i != predictions.size1();++i){
			RANGE_CHECK(target(i) <= 1);
			double prediction = predictions(i,0);
			error += evalErrorBinary(target(i),prediction);
			if ( target(i) == 0) {
				prediction -= 1.0; // multiply the argument to the logarithm by -1 to get the correct derivative for negative-class later (chain rule)
			}
			gradient(i,0) = -1.0/prediction;
		}
	}
	else {
		gradient.clear();
		for(std::size_t i = 0; i != predictions.size1();++i){
			RANGE_CHECK( target(i) < psize );
			double prediction = predictions(i,target(i));
			if ( prediction < m_minArgToLog ) {
				error += -m_minLogReturnVal;
			} else {
				error += -std::log(prediction);
			}
			if ( prediction == 0 ){
				prediction = m_minArgToLog;
			}
			gradient(i,target(i)) = -1.0/prediction;
		}
	}
	return error;
}
