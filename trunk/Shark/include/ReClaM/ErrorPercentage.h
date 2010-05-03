//===========================================================================
/*!
 *  \file ErrorPercentage.h
 * 
 *  \brief Calculates the error percentage based on the mean squared error.
 *
 *  \par Project:
 *      ReClaM
 *
 *  \par Copyright (c) 1999-2005:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
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

//____________________________________________________________________________________________________


#ifndef ERROR_PERCENTAGE_H
#define ERROR_PERCENTAGE_H


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>


//===========================================================================
/*!
 *  \brief Calculates the error percentage based on the mean squared error.
 *
 *  Measures the euklidian distance between the model output \em model(in),
 *  calculated from the input vector \em input, and the target vector \em target.
 *  Besides normalizing the squared error to the number \em P of patterns,
 *  it is also divided with the number of outputs \em N and the range
 *  of the target vectors. The resulting value is used to determine
 *  the percentage of errors.
 *  \f[
 *      E = \frac{100}{NP(max\{out_{ip}\} - min\{out_{ip}\})^{2}}
 *      \sum_{p=1}^P\sum_{i=1}^N(model(in)_{ip} -
 *      out_{ip})^{2}
 *  \f]
 *
 *      \param  model   The model used for prediction
 *      \param  input   Input vector for the model.
 *      \param  target  Target vector.
 *      \return The error percentage \em E.
 *
 *  \author  C. Igel
 * 
 */
class ErrorPercentage : public ErrorFunction
{
public:
	double error(Model& model, const Array<double>& input, const Array<double>& target)
	{
		double se = 0.0;
		double outmax = -MAXDOUBLE;
		double outmin = MAXDOUBLE;

		if (input.ndim() == 1)
		{
			Array<double> output(target.dim(0));
			model.model(input, output);
			for (unsigned c = 0; c < target.dim(0); c++)
			{
				if (target(c) > outmax) outmax = target(c);
				if (target(c) < outmin) outmin = target(c);
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
					if (target(pattern, c) > outmax) outmax = target(pattern, c);
					if (target(pattern, c) < outmin) outmin = target(pattern, c);
					se += (target(pattern, c) - output(c)) * (target(pattern, c) - output(c));
				}
			}
			se /= input.dim(0);
		}
		return 100.0 * se / (model.getOutputDimension() *(outmax - outmin) *(outmax - outmin));
	}
};


#endif

