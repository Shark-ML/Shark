//=================================================================================
/*!
*  \file    KalmanFilter.h
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
*     Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*  \par Project:
*      ReClaM
* 
*
*  <BR>
*
*
*  <BR><HR>
*  This file is part of ReClaM. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 2, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, write to the Free Software
*  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
*/
//=================================================================================


#ifndef _KALMANFILTER_H_
#define _KALMANFILTER_H_


#include <Array/Array2D.h>
#include <Array/ArrayOp.h>
#include <ReClaM/Model.h>


//!
//! The class provides an implementation of a standard linear KalmanFilter.
//!
//! The implementation aims at providing flexibility w.r.t. process and
//! measurement model; no restrictive assumptions concerning specific
//! properties of the matrices are made (although in practical applications
//! often noise covariances have diagonal form). The implementation is not
//! optimized at all for computational efficiency.
//!
//! For making the algorithm robust against numerical inaccuraries, the
//! update equation for the state error covariance matrix 'P' given in
//! "Applied Optimal Estimation; The Analytic Sciences Corporation; The M.I.T.
//! Press, pp. 305" is used. For very complex modells a square root form
//! (e.g. Cholesky) based implementation might be more appropriate due to
//! its increased numerical robustness. \par
//!
//! Process Modell: <br>
//! --------------- <br>
//! State Dynamics:   \f$  x_t = F*x_{t-1} + n_x;  n_x ~ N(0,Q)  \f$\n
//! Observation   :   \f$  z_t = H*x_t + n_z;      n_z ~ N(0,R)  \f$\n
//!
//!
//! Before filtering the model matrices F,Q,H,R must be set. Furthermore
//! the initial filter states, i.e. the initial state vector x(0) and the
//! initial covariance matrix of state estimation error P(0) must be set.
//!
//! As a Model subclass the KalmanFilter holds its current a-posteriori
//! state estimate at time t (after z(t) has been observed) in the
//! Model::parameter vector.
//!
class KalmanFilter : public Model
{
public:
	//! Constructor
	KalmanFilter();

	//! Destructor
	~KalmanFilter();


	//! The model first runs the predict phase.
	//! Then it updates its internal estimates
	//! using the input. Finally the a-posteriori
	//! state estimate, that is, the model
	//! parameter vector is returned as output.
	void model(const Array<double>& input, Array<double>& output);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of system process model
	//

	//! sets the state transition matrix 'F'
	void setStateDynamics(const Array2D<double>& state_transition);

	//! sets the state noise covariance matrix 'Q'
	void setStateNoise(const Array2D<double>& state_noise_cov);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of observation model
	//

	//! sets the observation matrix 'H'
	void setObservationModel(const Array2D<double>& observation_model);


	//! sets the observation noise covriance matrix 'R'
	void setObservationNoise(const Array2D<double>& observation_noise_cov);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// definition of initial state
	//

	//! sets the initial state vector (1-D array)
	void setStateVector(const Array<double>& state_vec);


	//! sets the initial covariance matrix of state estimation error
	void setStateErrorCov(const Array2D<double>& state_vec);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// prediction / filtering operations
	//

	//! next time step / forward propgation (calculation of a-priori estimates)
	void predict();

	//! update state vector; parameter z: measurent vector (1D-array)
	void updateEstimate(const Array<double>& z);

	//! update state vector; observation must have length equal to 'getMeasurementDim()'
	void updateEstimate(const double* observation);


	//--------------------------------------------------------------------------
	//--------------------------------------------------------------------------
	// access of data ...
	//

	//! returs a-priori (predicted) state estimate (updated in 'propagate()')
	inline Array<double>& get_x_a()
	{
		return x_a;
	}

	//! returs the a-posteriori (filtered) state estimate (updated in 'updateEstimate()')
	inline Array<double>& get_x_p()
	{
		// the model parameter vector represents the a-posteriori state
		return parameter;
	}

	//! returns the state transition matrix
	inline Array2D<double>& getF()
	{
		return F;
	}

	//! returns observation matrix
	inline Array2D<double>& getH()
	{
		return H;
	}

	//! returns a-priori (predicted) covariance of state estimation error
	inline Array2D<double>& getP_a()
	{
		return P_a;
	}

	//! returns a-posteriori covariance of state estimation error (after z(t) has been observed)
	inline Array2D<double>& getP_p()
	{
		return P_p;
	}

	//! returns Kalman gain matrix
	inline Array2D<double>& getK()
	{
		return K;
	}

	//! returns error in predicted measurements
	inline Array<double>& get_z_a_err()
	{
		return z_pred_err;
	}

	//! returns the state noise covariance matrix
	inline Array2D<double>& getQ()
	{
		return Q;
	}

	//! returns observation noise covariance matrix
	inline Array2D<double>& getR()
	{
		return R;
	}

	//! returns the dimension of the state vector
	inline int getStateDim() const
	{
		return F.dim(0);
	}

	//! returns the dimension of the measurement vector
	inline int getMeasurementDim() const
	{
		return H.dim(0);
	}

protected:
	//! state transition matrix
	Array2D<double> F;

	//! state noise covariance matrix
	Array2D<double> Q;

	//! observation matrix
	Array2D<double> H;

	//! observation noise covariance matrix
	Array2D<double> R;

	//! a-priori (predicted) state estimate at time t (before z(t) has been observed)
	Array<double> x_a;

	//! a-priori (predicted) covariance of state estimation error
	Array2D<double> P_a;

	//! a-posteriori covariance of state estimation error (after z(t) has been observed)
	Array2D<double> P_p;

	//! Kalman gain matrix
	Array2D<double> K;

	//! error in predicted measurements
	Array<double> z_pred_err;

	//! transpose of F; must be updated when F changes!
	Array2D<double> Ft;

	//! transpose of H; must be updated when H changes!
	Array2D<double> Ht;

	//! dimension of state vector
	int dimStateVec;
};


#endif

