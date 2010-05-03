//===========================================================================
/*!
*  \file KalmanFilterTest.cpp
*
*  \brief Implementation of a Wikipedia example of a linear Kalman filter
*
*  \author  T. Glasmachers
*  \date    2007
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
*  \par Project:
*      ReClaM
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



#include <stdio.h>
#include <stdlib.h>


#include <Rng/GlobalRng.h>
#include <ReClaM/KalmanFilter.h>


// implementation of the example from
// http://en.wikipedia.org/wiki/Kalman_filter
int main(int argc, char* argv[])
{
	Rng::seed(42);

	const double stddev_a = 10.0;			// standard deviation of acceleration
	const double stddev_z = 1.0;			// standard deviation of measurement
	const double delta_t = 0.04;

	const int x_dim  = 2;					// size of state vector
	const int z_dim  = 1;					// size of measurement vector

	Array2D<double> F(x_dim, x_dim);      	// state transition matrix
	Array2D<double> Q(x_dim, x_dim);      	// state noise covariance matrix
	Array2D<double> H(z_dim, x_dim);      	// observation matrix
	Array2D<double> R(z_dim, z_dim);      	// observation noise covariance matrix

	Array<double> xInit((unsigned) x_dim);	// initial state vector
	Array2D<double> PInit(x_dim, x_dim);  	// initial state error covariance matrix

	//--------------------------------------
	// define system model ...
	//
	F(0, 0) =  1.0;
	F(0, 1) =  delta_t;
	F(1, 0) =  0.0;
	F(1, 1) =  1.0;

	Q(0, 0) = 0.25 * pow(delta_t, 4) * stddev_a*stddev_a;
	Q(0, 1) = 0.5  * pow(delta_t, 3) * stddev_a*stddev_a;
	Q(1, 0) = 0.5  * pow(delta_t, 3) * stddev_a*stddev_a;
	Q(1, 1) =        pow(delta_t, 2) * stddev_a*stddev_a;

	//-------------------------------------------------------------------------------
	// define measurement model ...
	//
	H(0, 0) = 1;
	H(0, 1) = 0;

	R(0, 0) = stddev_z * stddev_z;

	//-------------------------------------------------------------------------------
	// define initial conditions
	//
	xInit = 0.0;							// initial state is (0, 0)
	PInit = 0.0;							// we are completely sure about this

	//-------------------------------------------------------------------------------
	// initialize KalmanFilter ...
	//
	KalmanFilter kalman;
	kalman.setStateDynamics(F);
	kalman.setStateNoise(Q);
	kalman.setObservationModel(H);
	kalman.setObservationNoise(R);
	kalman.setStateVector(xInit);
	kalman.setStateErrorCov(PInit);

	//-------------------------------------------------------------------------------
	// perform filtering of simulated 1D movement
	//
	Array<double> x(2);				// true current state
	Array<double> z(1);				// observation
	Array<double> prediction(2);	// kalman filter state prediction
	double a;						// true (random) acceleration
	x = 0.0;						// starting state
	int step;
	int steps = 10000;				// perform 10000 steps
	int output = 250;				// output every 250 steps

	printf("----------------------------------------------------------------------\n");
	printf("  location  \t  speed     \t  pred. loc.\t  pred. speed\n");
	printf("----------------------------------------------------------------------\n");
	for (step=0; step<=steps; step++)
	{
		// output true state and kalman filter prediction
		if ((step % output) == 0)
		{
			printf("%10.6g \t%10.6g \t%10.6g \t%10.6g\n", x(0), x(1), prediction(0), prediction(1));
		}

		// draw a random acceleration and update the true state
		a = Rng::gauss(0.0, stddev_a*stddev_a);
		Array<double> tmp = x;
		x(0) = tmp(0) + delta_t * tmp(1) + 0.5 * delta_t * delta_t * a;
		x(1) = tmp(1) + delta_t * a;
		z(0) = x(0) + Rng::gauss(0.0, stddev_z*stddev_z);

		// update the kalman filter and retreive its prediction
		// of the current state
		kalman.model(z, prediction);
	}
	printf("----------------------------------------------------------------------\n");

	return 0;
}

