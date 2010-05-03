//=================================================================================
/*!
*  \file    KalmanFilter.cpp
*
*  \brief   standard linear Kalman filter
*
*  \author  Thomas Bcher
*  \date    2004-11-07
* 
*  \par
*      This class is a port of the old cvKalmanFilter
*      class originally provided by the LinAlg package.
*
*  \par Copyright (c) 1999-2007:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project: ReClaM
* 
*
*  <BR>
*
*
*  <BR><HR>
*   This file is part of ReClaM. This library is free software;
*   you can redistribute it and/or modify it under the terms of the
*   GNU General Public License as published by the Free Software
*   Foundation; either version 2, or (at your option) any later version.
* 
*   This library is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*   GNU General Public License for more details.
* 
*   You should have received a copy of the GNU General Public License
*   along with this library; if not, write to the Free Software
*   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
//=================================================================================


#include <LinAlg/LinAlg.h>
#include <ReClaM/KalmanFilter.h>


KalmanFilter::KalmanFilter()
{
}

KalmanFilter::~KalmanFilter()
{
}


void KalmanFilter::model(const Array<double>& input, Array<double>& output)
{
	predict();
	updateEstimate(input);
	output = parameter;
}

void KalmanFilter::setStateDynamics(const Array2D<double>& state_transition)
{
	F = state_transition;
	Ft = transpose(F);
}

void KalmanFilter::setStateNoise(const Array2D<double>& state_noise_cov)
{
	Q = state_noise_cov;
}

void KalmanFilter::setObservationModel(const Array2D<double>& observation_model)
{
	H = observation_model;
	Ht = transpose(H);
}

void KalmanFilter::setObservationNoise(const Array2D<double>& observation_noise_cov)
{
	R = observation_noise_cov;
}

void KalmanFilter::setStateVector(const Array<double>& state_vec)
{
	x_a = state_vec;
	parameter = state_vec;

	dimStateVec = state_vec.dim(0);
}

void KalmanFilter::setStateErrorCov(const Array2D<double>& state_err_cov)
{
	P_a = state_err_cov;
	P_p = state_err_cov;
}

void KalmanFilter::predict()
{
	//------------------------------------------------------------
	// prediction of state vector and estimation error covariance
	//   parameter(t-1)  -->  x_a(t)
	//   P_p(t-1)  -->  P_a(t)
	x_a = innerProduct(F, parameter);                 // x_a(t) = F*parameter(t-1)
	P_a = innerProduct(F, innerProduct(P_p, Ft)) + Q; // P_a(t) = F*P_p(t-1)*F' + Q
}

void KalmanFilter::updateEstimate(const double* observation)
{
	Array<double> z(getMeasurementDim());
	for (int i = 0; i < getMeasurementDim(); ++i)
		z(i) = observation[i];

	updateEstimate(z);
}

void KalmanFilter::updateEstimate(const Array<double>& z)
{
	// error in predicted measurement
	z_pred_err = z - innerProduct(H, x_a);

	//------------------------------------------------------------
	// calculation of K(t) = P_a*H'*[H*P_a(t)*H' + R ]^(-1) "kalman gain"
	//
	Array2D<double> Kt1 = innerProduct(H, innerProduct(P_a, Ht)) + R;
	K = innerProduct(innerProduct(P_a, Ht), invert(Kt1));

	// a postiori state estimate
	parameter = x_a + innerProduct(K, z_pred_err);

	//------------------------------------------------------------
	// calculation of P_p(t) = (I-K(t)*H)P_a(t)*(I-K(t)*H)' + K(t)*R*K(t)'
	//
	// calculation of I-K(t)*H
	Array2D<double> ImKH = innerProduct(-K, H);

	for (int i = 0; i < dimStateVec; ++i)
		ImKH(i, i) += 1;

	Array<double> ImKHt = transpose(ImKH);
	Array<double> Kt    = transpose(K);
	P_p = innerProduct(ImKH, innerProduct(P_a, ImKHt)) + innerProduct(K, innerProduct(R, Kt));

	// in case that 'newMeasurement' is called repeatedly in one time step
	// (i.e. 'predict()' is not called between 'newMeasurement' calls).
	x_a = parameter;
	P_a = P_p;
}

