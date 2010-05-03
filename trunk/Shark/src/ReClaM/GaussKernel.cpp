//===========================================================================
/*!
 *  \file GaussKernel.cpp
 *
 *  \brief Gauss kernels with adaptive covariance matrices
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
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
//===========================================================================

#include <ReClaM/GaussKernel.h>


DiagGaussKernel::DiagGaussKernel(int dim, double gamma)
{
	// initialize D with the unit matrix
	parameter.resize(dim, false);
	parameter = gamma;
}

DiagGaussKernel::~DiagGaussKernel()
{
}


double DiagGaussKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	int i, dim = parameter.dim(0);

	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK((int)x1.dim(0) == dim);
	SIZE_CHECK((int)x2.dim(0) == dim);

	double a;
	double dist2 = 0.0;
	for (i = 0; i < dim; i++)
	{
		a = parameter(i) * (x1(i) - x2(i));
		dist2 += a * a;
	}
	return exp(-dist2);
}

double DiagGaussKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	int dim = parameter.dim(0);

	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK((int)x1.dim(0) == dim);
	SIZE_CHECK((int)x2.dim(0) == dim);

	derivative.resize(dim, false);
	int i;
	double a, b;
	double dist2 = 0.0;
	for (i = 0; i < dim; i++)
	{
		a = x1(i) - x2(i);
		b = parameter(i) * a;
		dist2 += b * b;
		derivative(i) = -2.0 * a * b;
	}
	double ret = exp(-dist2);
	for (i = 0; i < dim; i++) derivative(i) *= ret;
	return ret;
}

double DiagGaussKernel::computeGamma()
{
	double det = 1.0;
	int p, dim = getParameterDimension();
	for (p = 0; p < dim; p++) det *= parameter(p);
	return pow(det, 1.0 / dim);
}


////////////////////////////////////////////////////////////


GeneralGaussKernel::GeneralGaussKernel(int dim, double gamma)
{
	// initialize M with a multiple of the unit matrix
	parameter.resize(dim *(dim + 1) / 2, false);
	parameter = 0.0;
	double diag = sqrt(gamma);
	int i;
	for (i = 0; i < dim; i++) parameter(index(i, i)) = diag;
}

GeneralGaussKernel::GeneralGaussKernel(const Array2D<double>& symmetricTransformation)
{
	SIZE_CHECK(symmetricTransformation.ndim() == 2);
	unsigned int i, j;
	unsigned int dim = symmetricTransformation.dim(0);
	SIZE_CHECK( symmetricTransformation.dim(1) == dim );
	parameter.resize(dim *(dim + 1) / 2, false);
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j <= i; j++)
		{
			parameter(index(i, j)) = symmetricTransformation(i, j);
		}
	}
}

GeneralGaussKernel::~GeneralGaussKernel()
{
}


double GeneralGaussKernel::eval(const Array<double>& x1, const Array<double>& x2) const
{
	int dim = (int)floor(sqrt(2.0 * getParameterDimension()));

	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK((int)x1.dim(0) == dim);
	SIZE_CHECK((int)x2.dim(0) == dim);

	Array<double> d(dim);
	int i, j;
	double a;
	double dist2 = 0.0;
	for (i = 0; i < dim; i++) d(i) = x1(i) - x2(i);
	for (i = 0; i < dim; i++)
	{
		a = 0.0;
		for (j = 0; j < dim; j++)
		{
			a += d(j) * parameter(index(i, j));
		}
		dist2 += a * a;
	}
	return exp(-dist2);
}

double GeneralGaussKernel::evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const
{
	int dim = (int)floor(sqrt(2.0 * getParameterDimension()));

	SIZE_CHECK(x1.ndim() == 1);
	SIZE_CHECK(x2.ndim() == 1);
	SIZE_CHECK((int)x1.dim(0) == dim);
	SIZE_CHECK((int)x2.dim(0) == dim);

	derivative.resize(getParameterDimension(), false);
	Array<double> d(dim);
	Array<double> t(dim);
	int i, j;
	double a;
	double dist2 = 0.0;
	double ret;
	for (i = 0; i < dim; i++) d(i) = x1(i) - x2(i);
	for (i = 0; i < dim; i++)
	{
		a = 0.0;
		for (j = 0; j < dim; j++)
		{
			a += d(j) * parameter(index(i, j));
		}
		t(i) = a;
		dist2 += a * a;
	}
	ret = exp(-dist2);
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j < i; j++)
		{
			derivative(index(i, j)) = -2.0 * (d(i) * t(j) + d(j) * t(i)) * ret;
		}
		derivative(index(i, i)) = -2.0 * (d(i) * t(i)) * ret;
	}
	return ret;
}

void GeneralGaussKernel::getTransformation(Array<double>& trans)
{
	int dim = (int)floor(sqrt(2.0 * getParameterDimension()));
	int i, j;
	double p;
	trans.resize(dim, dim, false);
	for (i = 0; i < dim; i++)
	{
		for (j = 0; j <= i; j++)
		{
			p = parameter(index(i, j));
			trans(i, j) = p;
			trans(j, i) = p;
		}
	}
}

double GeneralGaussKernel::computeGamma()
{
	int dim = (int)floor(sqrt(2.0 * getParameterDimension()));
	double det;
	int i, j;
	double p;

	Array2D<double> M(dim, dim);
	Array2D<double> v(dim, dim);
	Array<double> d(dim);

	for (i = 0; i < dim; i++)
	{
		for (j = 0; j <= i; j++)
		{
			p = parameter(index(i, j));
			M(i, j) = p;
			M(j, i) = p;
		}
	}
	det = detsymm(M, v, d);

	return pow(det, 2.0 / dim);
}
