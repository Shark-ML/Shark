/*!
 *  \file EarlyStopping.cpp
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
 *   <BR>
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


#include <SharkDefs.h>
#include <ReClaM/EarlyStopping.h>


//===========================================================================
/*!
 *  \brief A new instance of EarlyStopping is created, internal variables
 *         are initialized.
 *
 *  The internal variables of the new instance are initialized and the
 *  time interval used for monitoring is set.
 *
 *  \param sl The number of time steps (striplength), for which the
 *            performance shall be evaluated. The default values is "5".
 *  \return none
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
EarlyStopping::EarlyStopping(unsigned sl = 5)
{
	GLvalue     = 0;
	e_opt       = MAXDOUBLE;
	e_va_old    = MAXDOUBLE;
	time        = 0;
	UPvalue     = 0;
	striplength = sl;
	e_tr.resize(striplength, false);
}

//===========================================================================
/*!
 *  \brief The performance characteristics are evaluated for the current
 *         time step.
 *
 *  Given the errors for the training and validation set for the current
 *  time step, the performance characteristics Generalization Loss,
 *  Trainings Progress and number of generalization loss increases
 *  are evaluated. When the current time step (= no. of method calls)
 *  is equal to the predefined striplength, the values are reset.
 *
 *  \param e    The current training set error.
 *  \param e_va The current validation set error.
 *  \return none
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
void EarlyStopping::update(double e, double e_va)
{
	double e_tr_av,   // Average training set error.
	e_tr_min;  // Minimal training set error.


	// evaluate generalization loss: GL
	if (e_va < e_opt)
		e_opt = e_va;
	GLvalue = 100. * (e_va / e_opt - 1.);

	// evaluate trainings-progress: TP
	if (time == 0) e_tr = e; // initialize error memory vector with first
	// error value
	// this implies a TPvalue == 1 when time == 0
	// this does not affect the TP(), see below
	e_tr(time % striplength) = e;
	e_tr_av = e_tr_min = e_tr(0);
	for (unsigned j = 1; j < striplength; j++)
	{
		e_tr_av += e_tr(j);
		if (e_tr(j) < e_tr_min)
			e_tr_min = e_tr(j);
	}
	e_tr_av /= (double)striplength;
	TPvalue = 1000. * (e_tr_av / e_tr_min - 1.);

	// evaluate increase of generalization error: UP
	if (time % striplength == 0)
	{
		if (e_va_old < e_va)
			UPvalue++;
		else
			UPvalue = 0;
		e_va_old = e_va;
	}

	// update time
	time++;
}


//===========================================================================
/*!
 *  \brief Returns the current value of the generalization loss.
 *
 *  \return The current generalization loss.
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
double EarlyStopping::getGL()
{
	return GLvalue;
}


//===========================================================================
/*!
 *  \brief Returns the current value of the training progress.
 *
 *  Call this method only if the current time step equals or exceeds
 *  the predefined striplength. Because of the calculation of the
 *  training progress you will get a value without significance
 *  otherwise.
 *
 *  \return The current generalization loss.
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
double EarlyStopping::getTP()
{
	if (time >= striplength)
		return TPvalue;
	else
		return MAXDOUBLE;
}

//===========================================================================
/*!
 *  \brief Returns the current value of the performance quotient.
 *
 *  Call this method only if the current time step equals or exceeds
 *  the predefined striplength. Because of the calculation of the
 *  performance quotient you will get a value without significance
 *  otherwise.
 *
 *  \return The current performance quotient.
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
double EarlyStopping::getPQ()
{
	if (time >= striplength)
		return GLvalue / TPvalue;
	else
		return 0.;
}


//===========================================================================
/*!
 *  \brief Returns the current number of generalization loss increases.
 *
 *  \return The current number of generalization loss increases.
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
unsigned EarlyStopping::getUP()
{
	return UPvalue;
}


//===========================================================================
/*!
 *  \brief Checks, whether the generalization loss exceeds a certain
 *         value.
 *
 *  \param alpha The value that is compared to the generalization loss.
 *               The default value is "1".
 *  \return "true", if the generalization loss exceeds \em alpha,
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
bool EarlyStopping::GL(double alpha = 1.0)
{
	return (GLvalue > alpha);
}


//===========================================================================
/*!
 *  \brief Checks, whether the trainings progress falls short of a certain
 *         value.
 *
 *  Call this method only if the current time step equals or exceeds
 *  the predefined striplength. Because of the calculation of the
 *  trainings progress you will get a value without significance
 *  otherwise.
 *
 *  \param alpha The value that is compared to the trainings progress
 *               The default value is "0.1".
 *  \return "true", if the trainings progress falls short of \em alpha,
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
bool EarlyStopping::TP(double alpha = 0.1)
{
	if (time >= striplength)
		return (TPvalue < alpha);
	else
		return false;
}

//===========================================================================
/*!
 *  \brief Checks, whether the performance quotient exceeds a certain
 *         value.
 *
 *  Call this method only if the current time step equals or exceeds
 *  the predefined striplength. Because of the calculation of the
 *  performance quotient you will get a value without significance
 *  otherwise.
 *
 *  \param alpha The value that is compared to the performance quotient
 *               The default value is "0.5".
 *  \return "true", if the performance progress exceeds \em alpha,
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
bool EarlyStopping::PQ(double alpha = 0.5)
{
	if (time >= striplength)
		return (GLvalue / TPvalue > alpha);
	else
		return false;
}


//===========================================================================
/*!
 *  \brief Checks, whether the number of generalization loss increases
 *         is equal to or greater than a certain value.
 *
 *  \param s The value that is compared to the number of generalization
 *           loss increases. The default value is "3".
 *  \return "true", if the number of generalization loss increases is equal
 *          to or greater than \em alpha, "false" otherwise.
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
bool EarlyStopping::UP(unsigned s = 3)
{
	return (UPvalue >= s);
}


//===========================================================================
/*!
 *  \brief Checks, whether one of the performance characteristics
 *         is out of range.
 *
 *  This method will check all performance characteristics against
 *  the given parameters:
 *
 *  <ul>
 *      <li>The generalization loss is out of range, when it is
 *          greater than \em alpha1
 *      <li>The trainings progress is out of range, when it is
 *          less than \em alpha2
 *      <li>The performance quotient is out of range, when it is
 *          greater than \em alpha3
 *      <li>The number of generalization loss increases is out of
 *          range, when it is equal to or greater than \em s
 *  </ul>
 *
 *  If at least one of the performance characteristics is out of range
 *  the method will return "true".
 *
 *  \param alpha1 The value that is compared to the generalization loss.
 *                The default value is "1".
 *  \param alpha2 The value that is compared to the trainings progress
 *                The default value is "0.1".
 *  \param alpha3 The value that is compared to the performance quotient
 *                The default value is "0.5".
 *  \param s      The value that is compared to the number of generalization
 *                loss increases. The default value is "3".
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
bool EarlyStopping::one_of_all(double alpha1 = 1.0, double alpha2 = 1.0, double alpha3 = 1.0, unsigned s = 3)
{
	return (GL(alpha1) || TP(alpha2) || PQ(alpha3) || UP(s));
}




