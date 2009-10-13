//===========================================================================
/*!
*  \file SigmoidModel.cpp
*
*  \brief sigmoidal functions
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 1999-2006:
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


#include <SharkDefs.h>
#include <ReClaM/SigmoidModel.h>


SigmoidModel::SigmoidModel(double A, double B)
{
	inputDimension = 1;
	outputDimension = 1;

	parameter.resize(2, false);
	parameter(0) = A;
	parameter(1) = B;
}

SigmoidModel::~SigmoidModel()
{
}


void SigmoidModel::model(const Array<double>& input, Array<double> &output)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		output.resize(1, false);
		output(0) = 1.0 / (1.0 + exp(parameter(0) * input(0) + parameter(1)));
	}
	else if (input.ndim() == 2)
	{
		int j, jc = input.dim(0);
		SIZE_CHECK(input.dim(1) == 1);
		output.resize(jc, 1, false);
		for (j = 0; j < jc; j++)
		{
			output(j, 0) = 1.0 / (1.0 + exp(parameter(0) * input(j, 0) + parameter(1)));
		}
	}
	else throw SHARKEXCEPTION("[SigmoidModel::model] invalid number of dimensions.");
}

void SigmoidModel::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		derivative.resize(1, 2, false);
		double e = exp(parameter(0) * input(0) + parameter(1));
		double f = 1.0 / (1.0 + e);
		double d = -f * f * e;
		derivative(0, 0) = d * input(0);
		derivative(0, 1) = d;
	}
	else throw SHARKEXCEPTION("[SigmoidModel::modelDerivative] invalid number of dimensions.");
}

void SigmoidModel::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		derivative.resize(1, 2, false);
		output.resize(1, false);
		double e = exp(parameter(0) * input(0) + parameter(1));
		double f = 1.0 / (1.0 + e);
		double d = -f * f * e;
		output(0) = f;
		derivative(0, 0) = d * input(0);
		derivative(0, 1) = d;
	}
	else throw SHARKEXCEPTION("[SigmoidModel::modelDerivative] invalid number of dimensions.");
}


////////////////////////////////////////////////////////////


SimpleSigmoidModel::SimpleSigmoidModel(double s)
{
	inputDimension = 1;
	outputDimension = 1;

	parameter.resize(1, false);
	parameter(0) = s;
}

SimpleSigmoidModel::~SimpleSigmoidModel()
{
}


void SimpleSigmoidModel::model(const Array<double>& input, Array<double> &output)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		output.resize(1, false);
		double x = parameter(0) * input(0);
		output(0) = 0.5 * x / (1.0 + fabs(x)) + 0.5;
	}
	else if (input.ndim() == 2)
	{
		int j, jc = input.dim(0);
		SIZE_CHECK(input.dim(1) == 1);
		output.resize(jc, 1, false);
		for (j = 0; j < jc; j++)
		{
			double x = parameter(0) * input(j, 0);
			output(j, 0) = 0.5 * x / (1.0 + fabs(x)) + 0.5;
		}
	}
	else throw SHARKEXCEPTION("[SimpleSigmoidModel::model] invalid number of dimensions.");
}

void SimpleSigmoidModel::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		derivative.resize(1, 1, false);
		double N = 1.0 + fabs(parameter(0) * input(0));
		derivative(0, 0) = 0.5 * input(0) / (N * N);
	}
	else throw SHARKEXCEPTION("[SimpleSigmoidModel::modelDerivative] invalid number of dimensions.");
}

void SimpleSigmoidModel::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == 1);
		derivative.resize(1, 1, false);
		output.resize(1, false);
		double x = parameter(0) * input(0);
		double N = 1.0 + fabs(x);
		output(0) = 0.5 * (x / N + 1.0);
		derivative(0, 0) = 0.5 * input(0) / (N * N);
	}
	else throw SHARKEXCEPTION("[SimpleSigmoidModel::modelDerivative] invalid number of dimensions.");
}

bool SimpleSigmoidModel::isFeasible()
{
	return (parameter(0) > 0.0);
}
