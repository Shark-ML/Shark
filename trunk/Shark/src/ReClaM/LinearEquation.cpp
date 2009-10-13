//===========================================================================
/*!
*  \file LinearEquation.cpp
*
*  \brief Model and Error Function for the iterative approximate
*         solution of a linear system
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


#include <ReClaM/LinearEquation.h>


LinearEquation::LinearEquation(const Array<double>& mat, const Array<double>& vec)
: matrix(mat)
, vector(vec)
{
	if (mat.ndim() != 2
			|| vec.ndim() != 1
			|| mat.dim(0) != vec.dim(0)) throw SHARKEXCEPTION("[LinearEquationError::LinearEquationError] dimension conflict");

	parameter.resize(mat.dim(1), false);
	parameter = 0.0;
}

LinearEquation::~LinearEquation()
{
}


void LinearEquation::model(const Array<double>& input, Array<double>& output)
{
	throw SHARKEXCEPTION("[LinearEquation::model] this is not a data processing model");
}

double LinearEquation::error()
{
	int i, ic = matrix.dim(0);
	int p, pc = matrix.dim(1);

	double ret = 0.0;
	for (i=0; i<ic; i++)
	{
		double sum = 0.0;
		for (p=0; p<pc; p++) sum += matrix(i, p) * getParameter(p);
		double diff = sum - vector(i);
		ret += diff * diff;
	}
	return ret;
}

double LinearEquation::errorDerivative(Array<double>& derivative)
{
	int i, ic = matrix.dim(0);
	int p, pc = matrix.dim(1);
	double ret = 0.0;
	derivative.resize(pc, false);
	Array<double> tmp(ic);
	for (i=0; i<ic; i++)
	{
		double sum = 0.0;
		for (p=0; p<pc; p++) sum += matrix(i, p) * getParameter(p);
		double diff = sum - vector(i);
		tmp(i) = 2.0 * diff;
		ret += diff * diff;
	}
	for (p=0; p<pc; p++)
	{
		double sum = 0.0;
		for (i=0; i<ic; i++) sum += matrix(i, p) * tmp(i);
		derivative(p) = sum;
	}
	return ret;
}

double LinearEquation::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	return error();
}

double LinearEquation::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	return errorDerivative(derivative);
}

void LinearEquation::getSolution(Array<double>& solution)
{
	solution = parameter;
}

