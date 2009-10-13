//===========================================================================
/*!
 *  \file DF_MeanSquaredError.h
 *
 *  \brief Mean Squared Error, using the GeneralDerivative interface
 *
 *  \par Copyright (c) 1998-2003:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      ReClaM
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

#ifndef DF_MEAN_SQUARED_ERROR__H
#define DF_MEAN_SQUARED_ERROR__H

#include <cmath>
#include "ReClaM/ErrorFunction.h"
#include "Array/ArrayOp.h"

//===========================================================================
/*!
 *   \brief Mean Squared Error, using the GeneralDerivative interface
 */
class DF_MeanSquaredError : public ErrorFunction
{
public:

//===========================================================================
	/*!
	 *  \brief Calculates the mean squared error between the output and the target
	 *         vector.
	 *
	 *  Measures the euklidian distance between the model output \em model(in),
	 *  calculated from the input vector \em in, and the target vector \em target.
	 *  The result is then normalized to the number of output neurons.
	 *  Consider the case of a N-dimensional output vector, i.e. a neural network
	 *  with \em N output neurons, and a set of \em P patterns. In this case the
	 *  function calculates
	 *  \f[
	 *      E = \frac{1}{NP} \sum_{p=1}^P\sum_{i=1}^N(model(output)_{i}^{p} -
	 *      target_{i}^{p})^{2}
	 *  \f]
	 *
	 *      \param  input Input vector for the model.
	 *      \param  target Target vector.
	 *      \return The mean squared error \em E.
	 *
	 *  \author  B. Sendhoff
	 *  \date    1999
	 *
	 *  \par Changes
	 *      C. Igel, M. Toussaint, 2001-09-13:<BR>
	 *      Normalising now with no. of output neurons
	 *      and no. of patterns.
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double error(Model& model, const Array<double> &input, const Array<double> &target)
	{
		double se = 0;
		if (input.ndim() == 1)
		{
			Array<double> output(target.dim(0));
			model.model(input, output);
			for (unsigned c = 0; c < target.dim(0); c++)
			{
				se += (target(c) - output(c)) * (target(c) - output(c));
			}
		}
		else
		{
			Array<double> output(target.dim(1));
			for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
			{
				model.model(input[pattern], output);
				for (unsigned c = 0; c < target.dim(1); c++)
				{
					se += (target(pattern, c) - output(c)) * (target(pattern, c) - output(c));
				}
			}
		}
		// normalise
		se /= target.nelem();
		return se;
	}

//===========================================================================
	/*!
	 *  \brief Calculates the derivatives of the mean squared error with respect
	 *         to the parameters ModelInterface::w.
	 *
	 *  According to the equation in the description for the function
	 *  #error the derivatives of the mean squared error can be calculated
	 *  The results are written to the vector ModelInterface::dedw.
	 *
	 *  Usually, as a byproduct of the calculation of the derivative one gets the
	 *  the error \f$E\f$ itself very efficiently. Therefore, the method #error
	 *  gives back this value. This additional effect can be switched of by means
	 *  of the third parameter (returnError = false).
	 *
	 *      \param  input Input vector for the model.
	 *      \param  target The target vector.
	 *      \param  returnError Determines whether or not to calculate the error
	 *                          itself. By default the error is calculated.
	 *      \return The error \em E if \em returnError is set to "true", "-1"
	 *              otherwise.
	 *
	 *  \author  B. Sendhoff
	 *  \date    1999
	 *
	 *  \par Changes
	 *      C. Igel, M. Toussaint, 2001-09-13:<BR>
	 *      Normalising now with no. of output neurons and
	 *      no. of patterns.
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	double errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative)
	{

		derivative = 0;
		derivative.resize(model.getParameterDimension(), false);

		double se = 0;

		if (input.ndim() == 1)
		{
			output.resize(target.dim(0));
			locerr.resize(output.nelem(), false);
			model.model(input, output);
			for (unsigned c = 0; c < output.nelem(); c++)
			{
				se += (target(c) - output(c)) * (target(c) - output(c));
				locerr(c) = (target(c) - output(c));
			}
			model.generalDerivative(input, locerr, derivative);
		}
		else
		{

			output.resize(target.dim(1));
			locerr.resize(output.nelem(), false);


			for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
			{
				model.model(input[pattern], output);

				for (unsigned c = 0; c < output.nelem(); c++)
				{
					se += (target(pattern, c) - output(c)) * (target(pattern, c) - output(c));
					locerr(c) = (target(pattern, c) - output(c));
				}
				model.generalDerivative(input, locerr, derivative);
			}
		}
		// normalise
		derivative *= 2. / target.nelem();
		se /= target.nelem();
		return se;
	}
	Array<double> output;
	Array<double> locerr;
};

#endif








