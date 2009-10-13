//===========================================================================
/*!
 *  \file EarlyStopping.h
 *
 *  \brief Used for monitoring purposes, to avoid overfitting.
 *
 *  When training neural networks, there is always the danger of
 *  overfitting, i.e. the prediction error for a training set
 *  is monotonically decreasing with the number of optimization
 *  algorithm iterations, but the error for a validation set
 *  (i.e. independent data) is first decreasing, followed by an
 *  increase. To achieve best generalization performance, the
 *  training should then be stopped at an iteration step when
 *  the validation set error has its minimum. <br>
 *  This file offers a class that can be used to monitor
 *  the performance of a network related to a couple of properties
 *  (e.g. generalization loss), so one can decide when to
 *  stop the training.
 *
 *  \author  M. H&uuml;sken
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2000:
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
//===========================================================================

#ifndef EARLYSTOPPING_H
#define EARLYSTOPPING_H

#include "Array/Array.h"


//===========================================================================
/*!
 *  \brief Used for monitoring purposes, to avoid overfitting.
 *
 *  When training neural networks, there is always the danger of
 *  overfitting, i.e. the prediction error for a training set
 *  is monotonically decreasing with the number of optimization
 *  algorithm iterations, but the error for a validation set
 *  (i.e. independent data) is first decreasing, followed by an
 *  increase. To achieve best generalization performance, the
 *  training should then be stopped at an iteration step when
 *  the validation set error has its minimum. <br>
 *  This class offers methods, that can be used to monitor
 *  the performance of a network related to a couple of performance
 *  characteristics, so one can decide when to stop the training. <br>
 *  The given performance characteristics are: <br>
 *
 *  The training progress (TP) given as
 *  \f$1000 \ast \frac{E_{T_{average}}}{E_{T_{min}} - 1}\f$, where
 *  \f$E_{T_{average}}\f$ is the average training set error evaluated
 *  over a period of timesteps (striplength) and \f$E_{T_{min}}\f$
 *  is the minimal error for the training set during this period. <br> <br>
 *
 *  The generalization loss (GL) given as
 *  \f$100 \ast \frac{E_{V}}{E_{V_{min}} - 1}\f$, where \f$E_{V}\f$ is the
 *  current error of the validation set and \f$E_{V_{min}}\f$ is the minimum
 *  error of the validation set, as calculated during the striplength.
 *  <br> <br>
 *
 *  The number of generalization loss increases (UP) is given
 *  as the number of time steps during the striplength, at which
 *  the generalization loss increases compared to the previous
 *  time step. <br> <br>
 *
 *  The performance quotient (PQ) given as
 *  \f$\frac{GL}{TP}\f$.
 *
 *  \par Example
 *  \code
 *  void main()
 *  {
 *
 *      // Create net:
 *      MyNet net;
 *
 *      // Fill net:
 *      ...
 *
 *      // Create early stopping instance for monitoring 5 time steps:
 *      EarlyStopping estop( 5 );
 *
 *      // Training the net:
 *      for ( unsigned epoch = 1; epoch < max_epochs; epoch++ )
 *      {
 *          // Train net with prefered optimization algorithm:
 *          ...
 *
 *          // Calculate training and validation error of
 *          // current model:
 *          ...

 *          // Update of monitoring variables:
 *          estop.update( error_train, error_validation );
 *
 *          // Overfitting of network? Then stop training:
 *          if ( estop.one_of_all( 1.0, 1.0, 1.0, 3) ) break;
 *
 *      } // next training epoch
 *  }
 *  \endcode
 *
 *  \author  M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 */
class EarlyStopping
{
public:


	//! A new instance of EarlyStopping is created, internal variables
	//! are initialized.
	EarlyStopping(unsigned);

	//! Returns the current value of the generalization loss.
	double getGL();

	//! Returns the current value of the training progress.
	double getTP();

	//! Returns the current value of the performance quotient.
	double getPQ();

	//! Returns the current number of generalization loss increases.
	unsigned getUP();

	//! Checks, whether the generalization loss exceeds a certain
	//! value.
	bool GL(double);


//===========================================================================
	/*!
	 *  \brief Checks, whether the generalization loss exceeds "1".
	 *         value.
	 *
	 *  \return "true", if the generalization loss exceeds "1",
	 *          "false" otherwise.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool GL();


	//! Checks, whether the trainings progress falls short of a certain
	//! value.
	bool TP(double);


//===========================================================================
	/*!
	 *  \brief Checks, whether the trainings progress falls short of "0.1".
	 *
	 *  Call this method only if the current time step equals or exceeds
	 *  the predefined striplength. Because of the calculation of the
	 *  trainings progress you will get a value without significance
	 *  otherwise.
	 *
	 *  \return "true", if the trainings progress falls short of "0.1",
	 *          "false" otherwise.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool TP();


	//! Checks, whether the performance quotient exceeds a certain
	//! value.
	bool PQ(double);


//===========================================================================
	/*!
	 *  \brief Checks, whether the performance quotient exceeds "0.5".
	 *
	 *  Call this method only if the current time step equals or exceeds
	 *  the predefined striplength. Because of the calculation of the
	 *  performance quotient you will get a value without significance
	 *  otherwise.
	 *
	 *  \return "true", if the performance quotient exceeds "0.5",
	 *          "false" otherwise.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool PQ();


	//! Checks, whether the number of generalization loss increases
	//! is equal to or greater than a certain value.
	bool UP(unsigned);


//===========================================================================
	/*!
	 *  \brief Checks, whether the number of generalization loss increases
	 *         is equal to or greater than "3".
	 *
	 *  \return "true", if the number of generalization loss increases is equal
	 *          to or greater than "3", "false" otherwise.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool UP();


	//! Checks, whether one of the performance characteristics
	//! is out of range.
	bool one_of_all(double, double, double, unsigned);


//===========================================================================
	/*!
	 *  \brief Checks, whether one of the performance characteristics
	 *         is out of default range.
	 *
	 *  This method will check all performance characteristics:
	 *
	 *  <ul>
	 *      <li>The generalization loss is out of range, when it is
	 *          greater than "1"</li>
	 *      <li>The trainings progress is out of range, when it is
	 *          less than "0.1"</li>
	 *      <li>The performance quotient is out of range, when it is
	 *          greater than "0.5"</li>
	 *      <li>The number of generalization loss increases is out of
	 *          range, when it is equal to or greater than "3"</li>
	 *  </ul>
	 *
	 *  If at least one of the performance characteristics is out of range
	 *  the method will return "true".
	 *
	 *  \return "true", if at least one of the performance characteristics is out
	 *          of range, "false" otherwise.
	 *
	 *  \author  M. H&uuml;sken
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	bool one_of_all();


	//! The performance characteristics are evaluated for the current
	//! time step.
	void update(double, double);


private:


	double GLvalue,   // The Generalization Loss value.
	TPvalue,   // The Trainings Progress value.
	e_opt,     // The optimal error.
	e_va_old;  // The old validation set error.

	Array<double> e_tr; // Memory for the training set error
	// saved for a number of time steps.


	unsigned striplength, // The number of time steps, for which
	// the training set errors shall be stored.
	time,        // The current time step.
	UPvalue;     // The increase of generalization error

};

#endif

