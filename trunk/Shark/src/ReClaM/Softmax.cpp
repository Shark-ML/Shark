//===========================================================================
/*!
*  \file Softmax.cpp
*
*  \brief soft-max function
*
*  \author  T. Glasmachers
*  \date    2010
*
*  \par Copyright (c) 1999-2010:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#include <ReClaM/Softmax.h>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>


Softmax::Softmax(unsigned int dim)
{
	inputDimension = dim;
	outputDimension = dim;

	unsigned int i;
	parameter.resize(2 * dim, false);
	for (i=0; i<dim; i++)
	{
		parameter(2*i) = -1.0;
		parameter(2*i+1) = 0.0;
	}
}

Softmax::~Softmax()
{
}


void Softmax::model(const Array<double>& input, Array<double> &output)
{
	unsigned int i;
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == inputDimension);
		output.resize(inputDimension, false);
		double sum = 0.0;
		for (i=0; i<inputDimension; i++)
		{
			double v = exp(parameter(2*i) * input(i) + parameter(2*i+1));
			output(i) = v;
			sum += v;
		}
		output /= sum;
	}
	else if (input.ndim() == 2)
	{
		SIZE_CHECK(input.dim(1) == inputDimension);
		int j, jc = input.dim(0);
		output.resize(jc, inputDimension, false);
		for (j = 0; j < jc; j++)
		{
			double sum = 0.0;
			for (i=0; i<inputDimension; i++)
			{
				double v = exp(parameter(2*i) * input(j, i) + parameter(2*i+1));
				output(j, i) = v;
				sum += v;
			}
			output[j] /= sum;
		}
	}
	else throw SHARKEXCEPTION("[Softmax::model] invalid number of dimensions.");
}

void Softmax::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == inputDimension);
		derivative.resize(inputDimension, 2 * inputDimension, false);
		unsigned int i, j;
		Array<double> Z(inputDimension);
		double N = 0.0;
		for (i=0; i<inputDimension; i++)
		{
			double v = exp(parameter(2*i) * input(i) + parameter(2*i+1));
			Z(i) = v;
			N += v;
		}
		for (j=0; j<inputDimension; j++)
		{
			double o = Z(j) / N;
			for (i=0; i<inputDimension; i++)
			{
				double t = -Z(i) / N;
				if (i == j) t += 1.0;
				t *= o;
				derivative(j, 2*i) = input(j) * t;
				derivative(j, 2*i+1) = t;
			}
		}
	}
	else throw SHARKEXCEPTION("[Softmax::modelDerivative] invalid number of dimensions.");
}

void Softmax::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		SIZE_CHECK(input.dim(0) == inputDimension);
		derivative.resize(inputDimension, 2 * inputDimension, false);
		output.resize(inputDimension, false);
		unsigned int i, j;
		Array<double> Z(inputDimension);
		double N = 0.0;
		for (i=0; i<inputDimension; i++)
		{
			double v = exp(parameter(2*i) * input(i) + parameter(2*i+1));
			Z(i) = v;
			N += v;
		}
		for (j=0; j<inputDimension; j++)
		{
			double o = Z(j) / N;
			for (i=0; i<inputDimension; i++)
			{
				double t = -Z(i) / N;
				if (i == j) t += 1.0;
				t *= o;
				derivative(j, 2*i) = input(i) * t;
				derivative(j, 2*i+1) = t;
			}
			output(j) = o;
		}
	}
	else throw SHARKEXCEPTION("[Softmax::modelDerivative] invalid number of dimensions.");
}
