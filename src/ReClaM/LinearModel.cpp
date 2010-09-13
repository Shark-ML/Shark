//===========================================================================
/*!
*  \file LinearModel.cpp
*
*  \brief Linear models on a real vector space
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
*  The #LinearFunction class provides a simple model, that is,
*  a linear function on a real vector space (a map to the reals).
*  The #LinearMap class provides a linear map from one real
*  vector space to another.
*
*
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


#include <math.h>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>
#include <ReClaM/LinearModel.h>


LinearMap::LinearMap(int inputdim, int outputdim)
{
	inputDimension = inputdim;
	outputDimension = outputdim;

	parameter.resize(inputdim * outputdim, false);
	parameter = 0.0;
}

LinearMap::~LinearMap()
{
}


void LinearMap::model(const Array<double>& input, Array<double> &output)
{
	if (input.ndim() == 1)
	{
		unsigned int i, o;
		double value;
		output.resize(outputDimension, false);
		int p = 0;
		for (o = 0; o < outputDimension; o++)
		{
			value = 0.0;
			for (i = 0; i < inputDimension; i++)
			{
				value += input(i) * parameter(p);
				p++;
			}
			output(o) = value;
		}
	}
	else if (input.ndim() == 2)
	{
		unsigned int j, jc = input.dim(0);
		unsigned int i, o;
		double value;
		output.resize(jc, outputDimension, false);
		for (j = 0; j < jc; j++)
		{
			int p = 0;
			for (o = 0; o < outputDimension; o++)
			{
				value = 0.0;
				for (i = 0; i < inputDimension; i++)
				{
					value += input(j, i) * parameter(p);
					p++;
				}
				output(j, o) = value;
			}
		}
	}
	else throw SHARKEXCEPTION("[LinearMap::model] invalid number of dimensions.");
}

void LinearMap::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		derivative.resize(outputDimension, getParameterDimension(), false);
		derivative=0.0;
		for (size_t o = 0; o < outputDimension; o++)
		{
			for (size_t i = 0; i < inputDimension; i++)
			{
				derivative(o, i+o*inputDimension) = input(i);
			}
		}
	}
	else throw SHARKEXCEPTION("[LinearMap::modelDerivative] invalid number of dimensions.");
}


////////////////////////////////////////////////////////////


AffineLinearMap::AffineLinearMap(int inputdim, int outputdim)
{
	inputDimension = inputdim;
	outputDimension = outputdim;

	parameter.resize(inputdim * outputdim + outputdim, false);
	parameter = 0.0;
}

AffineLinearMap::~AffineLinearMap()
{
}


void AffineLinearMap::model(const Array<double>& input, Array<double> &output)
{
	int pb = inputDimension * outputDimension;

	if (input.ndim() == 1)
	{
		unsigned int i, o;
		double value;
		output.resize(outputDimension, false);
		int p = 0;
		for (o = 0; o < outputDimension; o++)
		{
			value = parameter(pb + o);
			for (i = 0; i < inputDimension; i++)
			{
				value += input(i) * parameter(p);
				p++;
			}
			output(o) = value;
		}
	}
	else if (input.ndim() == 2)
	{
		unsigned int j, jc = input.dim(0);
		unsigned int i, o;
		double value;
		output.resize(jc, outputDimension, false);
		for (j = 0; j < jc; j++)
		{
			int p = 0;
			for (o = 0; o < outputDimension; o++)
			{
				value = parameter(pb + o);
				for (i = 0; i < inputDimension; i++)
				{
					value += input(j, i) * parameter(p);
					p++;
				}
				output(j, o) = value;
			}
		}
	}
	else throw SHARKEXCEPTION("[AffineLinearMap::model] invalid number of dimensions.");
}

void AffineLinearMap::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	if (input.ndim() == 1)
	{
		size_t parameterPerOutput=inputDimension+1;
		derivative.resize(outputDimension, getParameterDimension(), false);
		derivative=0.0;
		for (size_t o = 0; o < outputDimension; o++)
		{
			for (size_t i = 0; i < inputDimension; i++)
			{
				derivative(o, i+o*parameterPerOutput) = input(i);
			}
			derivative(o, o*parameterPerOutput+inputDimension) = 1;
		}
	}
	else throw SHARKEXCEPTION("[AffineLinearMap::modelDerivative] invalid number of dimensions.");
}


////////////////////////////////////////////////////////////


LinearFunction::LinearFunction(int dimension)
: LinearMap(dimension, 1)
{
}


////////////////////////////////////////////////////////////


AffineLinearFunction::AffineLinearFunction(int dimension)
: AffineLinearMap(dimension, 1)
{
}


////////////////////////////////////////////////////////////


LinearClassifier::LinearClassifier(int dimension, int classes)
{
	inputDimension = dimension;
	outputDimension = classes;
	numberOfClasses = classes;

	// parameters are class mean vectors and global covariance matrix
	mean.resize(classes, dimension, false);
	covariance.resize(dimension, dimension, false);
	inverse.resize(dimension, dimension, false);
	parameter.resize(dimension * classes + dimension * (dimension+1) / 2, false);
	mean = 0.0;
	covariance = 0.0;
	inverse = 0.0;
	parameter = 0.0;

	bNeedsUpdate = true;
}

LinearClassifier::~LinearClassifier()
{
}


void LinearClassifier::setParameter(unsigned int index, double value)
{
	Model::setParameter(index, value);

	if (index < numberOfClasses * inputDimension)
	{
		// mean vector component
		int cls = index / inputDimension;
		int dim = index % inputDimension;
		mean(cls, dim) = value;
	}
	else
	{
		// symmetric matrix component
		index -= numberOfClasses * inputDimension;
		int y = (int)floor(sqrt(2.0*index + 0.25) - 0.5);
		int x = index - y*(y+1)/2;
		covariance(x, y) = covariance(y, x) = value;
		bNeedsUpdate = true;
	}
}

void LinearClassifier::model(const Array<double>& input, Array<double>& output)
{
	if (bNeedsUpdate)
	{
		invertSymm(inverse, covariance);
		bNeedsUpdate = false;
	}

	Array<double> diff(inputDimension);

	if (input.ndim() == 1)
	{
		output.resize(numberOfClasses, false);
		output = 0.0;
		int c, d;
		int best = 0;
		double dist2, bestDist = MAXDOUBLE;
		for (c=0; c<numberOfClasses; c++)
		{
			for (d=0; d<(int)inputDimension; d++) diff(d) = input(d) - mean(c, d);
			dist2 = vecMatVec(diff, inverse, diff);
			if (dist2 < bestDist)
			{
				bestDist = dist2;
				best = c;
			}
		}
		output(best) = 1.0;
	}
	else if (input.ndim() == 2)
	{
		int i, ic = input.dim(0);
		output.resize(ic, numberOfClasses, false);
		output = 0.0;
		for (i=0; i<ic; i++)
		{
			int c, d;
			int best = 0;
			double dist2, bestDist = MAXDOUBLE;
			for (c=0; c<numberOfClasses; c++)
			{
				for (d=0; d<(int)inputDimension; d++) diff(d) = input(i, d) - mean(c, d);
				dist2 = vecMatVec(diff, inverse, diff);
				if (dist2 < bestDist)
				{
					bestDist = dist2;
					best = c;
				}
			}
			output(i, best) = 1.0;
		}
	}
	else throw SHARKEXCEPTION("[LinearClassifier::model] invalid number of dimensions.");
}
