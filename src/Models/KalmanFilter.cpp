//===========================================================================
/*!
 *  \file KalmanFilter.cpp
 *
 *  \brief KalmanFilter
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
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
 *
 */
//===========================================================================
#include <shark/LinAlg/Inverse.h>
#include <shark/Models/KalmanFilter.h>

using namespace shark;

KalmanFilter::KalmanFilter(size_t modelSize){
	m_inputSize=modelSize;
	m_name="KalmanFilter";
}

void KalmanFilter::eval(const RealVector& pattern,RealVector& output){
	predict();
	updateEstimate(pattern);
	output = m_state;
}
void KalmanFilter::setStateVector(const RealVector& stateVec){
	m_predictedState = stateVec;
	m_state = stateVec;
}

void KalmanFilter::setStateErrorCov(const RealMatrix& state_err_cov){
	m_predictedCovariance = state_err_cov;
	m_observedCovariance = state_err_cov;
}

void KalmanFilter::predict(){
	//------------------------------------------------------------
	// prediction of state vector and estimation error covariance
	//   parameter(t-1)  -->  x_a(t)
	//   P_p(t-1)  -->  P_a(t)
	m_predictedState = prod(m_stateTransition, m_state);// x_a(t) = F*parameter(t-1)
	RealMatrix P_temp=prod(m_observedCovariance, trans(m_stateTransition));
	m_predictedCovariance = prod(m_stateTransition, P_temp) + m_stateNoiseCov; // P_a(t) = F*P_p(t-1)*F' + Q
}


void KalmanFilter::updateEstimate(const RealVector& z){
	// error in predicted measurement
	m_predictionError = z - prod(m_observationModel, m_predictedState);

	//------------------------------------------------------------
	// calculation of K(t) = P_a*H'*[H*P_a(t)*H' + R ]^(-1) "kalman gain"
	//
	RealMatrix KT1_temp=prod(m_predictedCovariance, trans(m_observationModel));
	RealMatrix Kt1 = prod(m_observationModel, KT1_temp) + m_observationNoise;
	RealMatrix K = prod(KT1_temp, invert(Kt1));

	// a postiori state estimate
	m_state += prod(K, m_predictionError);

	//------------------------------------------------------------
	// calculation of P_p(t) = (I-K(t)*H)P_a(t)*(I-K(t)*H)' + K(t)*R*K(t)'
	//
	// calculation of I-K(t)*H
	RealMatrix ImKH = RealIdentity(m_inputSize)-prod(K, m_observationModel);

	RealMatrix P_temp1=prod(m_predictedCovariance,trans(ImKH));
	RealMatrix R_Kt= prod(m_observationNoise, trans(K));
	m_observedCovariance = prod(ImKH, P_temp1) + prod(trans(K), R_Kt);

	// in case that 'newMeasurement' is called repeatedly in one time step
	// (i.e. 'predict()' is not called between 'newMeasurement' calls).
	m_predictedState = m_state;
	m_predictedCovariance = m_observedCovariance;
}

