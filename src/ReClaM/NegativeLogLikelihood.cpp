//===========================================================================
/*!
 *  \file NegativeLogLikelihood.cpp
 *
 *  \brief negative logarithm of the likelihood of a probabilistic binary classification model
 *
 *  \author  T. Glasmachers
 *  \date    2008
 *
 *  \par Copyright (c) 1999-2008:
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


#include <ReClaM/NegativeLogLikelihood.h>


NegativeLogLikelihood::NegativeLogLikelihood()
{
}

NegativeLogLikelihood::~NegativeLogLikelihood()
{
}


double NegativeLogLikelihood::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	double ret = 0.0;
	if (input.ndim() == 1)
	{
		Array<double> output(1);
		model.model(input, output);
		if (target(0) > 0.0) ret -= log(output(0));
		else ret -= log(1.0 - output(0));
	}
	else if (input.ndim() == 2)
	{
		int i, ic = input.dim(0);
		Array<double> output(ic, 1);
		model.model(input, output);
		for (i=0; i<ic; i++)
		{
			if (target(i, 0) > 0.0) ret -= log(output(i, 0));
			else ret -= log(1.0 - output(i, 0));
		}
	}
	else throw SHARKEXCEPTION("[NegativeLogLikelihood::error] invalid dimension");
	return ret;
}

double NegativeLogLikelihood::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	double ret = 0.0;
	int p, pc = model.getParameterDimension();
	derivative.resize(pc, false);
	derivative = 0.0;
	Array<double> output(1);
	Array<double> dmdw;
	if (input.ndim() == 1)
	{
		model.modelDerivative(input, output, dmdw);
		double o = output(0);
		if (target(0) > 0.0)
		{
			ret -= log(o);
		}
		else
		{
			ret -= log(1.0 - o);
			o -= 1.0;
		}
		for (p=0; p<pc; p++) derivative(p) -= dmdw(0, p) / o;
	}
	else if (input.ndim() == 2)
	{
		Array<double> der(pc);
		int i, ic = input.dim(0);
		for (i=0; i<ic; i++)
		{
			model.modelDerivative(input[i], output, dmdw);
			double o = output(0);
			if (target(i, 0) > 0.0)
			{
				ret -= log(o);
			}
			else
			{
				ret -= log(1.0 - o);
				o -= 1.0;
			}
			for (p=0; p<pc; p++) derivative(p) -= dmdw(0, p) / o;
		}
	}
	else throw SHARKEXCEPTION("[NegativeLogLikelihood::errorDerivative] invalid dimension");
	return ret;
}
