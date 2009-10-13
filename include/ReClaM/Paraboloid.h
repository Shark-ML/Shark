//===========================================================================
/*!
 *  \file Paraboloid.h
 *
 *  \brief Convex quadratic model and error function
 * 
 *  \author C. Igel
 *
 *  \par Copyright (c) 1998-2004:
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

//=============================================================
//
// paraboloid test function
//

#ifndef PARABOLOID_H
#define PARABOLOID_H

#include <SharkDefs.h>
#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>
#include <Rng/GlobalRng.h>
#include <vector>
#include <Array/ArrayOp.h>
#include <Array/ArrayIo.h>


//!
//! \brief Convex quadratic model and error function
//!
//! \par
//! This model/error function is regularly used to test
//! the behavior of optimizers.
//!
class Paraboloid : public Model, public ErrorFunction
{
public:
	Paraboloid(unsigned _d, double _c = 10, bool basis = true)
	{
		unsigned i;
		d = _d;
		cond = _c;

		parameter.resize(d);
		parameter = 0.0;
		B   .resize(d, d);
		if (basis) generateBasis();
		else
		{
			B  = 0;
			for (i = 0; i < d; i++) B(i, i) = 1;
		}
	}

	void model(const Array<double> &input, Array<double> &output)
	{
	}

	void modelDerivative(const Array<double> &input, Array<double>& derivative)
	{
	}

	double error(const std::vector<double> &v)
	{
		unsigned i;
		if (v.size() != d)
		{
			throw SHARKEXCEPTION("dimension mismatch");
		}
		double sum = 0.;
		for (i = 0; i < d; i++) parameter(i) = v[i];
		for (i = 0; i < d; i++)
		{
			sum += Shark::sqr(pow(cond, (double(i) / double(d - 1))) * scalarProduct(parameter, B.col(i))) ;
		}
		return sum;
	}

	double error(Model& model, const Array<double> &input, const Array<double> &target)
	{
		double sum = 0.;
		unsigned i;
		for (i = 0; i < d; i++)
			sum += Shark::sqr(pow(cond, double(i) / double(d - 1)) * scalarProduct(parameter, B.col(i)));
		return sum;
	}

	double errorDerivative(Model& model, const Array<double> &input, const Array<double> &target, Array<double>& derivative)
	{
		double sum = 0.;
		unsigned k, i;

		derivative.resize(d, false);
		derivative = 0.0;
		for (k = 0; k < d; k++)
		{
			sum += Shark::sqr(pow(1.0, (double) k / (double)(d - 1)) * scalarProduct(parameter, B.col(k)));
			for (i = 0; i < d; i++)
				derivative(k) += 2.0 * pow(cond, 2.0 * double(i) / double(d - 1)) * scalarProduct(parameter, B.col(i)) * B(k, i);
		}
		return sum;
	}

	void init(unsigned s, double a, double b, bool basis = true)
	{
		unsigned i;
		Rng::seed(s);
		if (basis) generateBasis();
		for (i = 0; i < d; i++) parameter(i) = Rng::uni(a, b);
	}

	Array<double> &getBasis()
	{
		return B;
	}

protected:
	void generateBasis()
	{
		unsigned i, j, c;
		Array<double> H;
		B.resize(d, d);
		H.resize(d, d);
		for (i = 0; i < d; i++)
		{
			for (c = 0; c < d; c++)
			{
				H(i, c) = Rng::gauss(0, 1);
				//H(i, c) = (c==i) ? 1 : 0;
			}
		}
		B = H;
		for (i = 0; i < d; i++)
		{
			for (j = 0; j < i; j++)
				for (c = 0; c < d; c++)
					B(i, c) -= scalarProduct(H[i], H[j]) * H(j, c) / scalarProduct(H[j], H[j]);
			H = B;
		}
		for (i = 0; i < d; i++)
		{
			double normB = sqrt(scalarProduct(B[i], B[i]));
			for (j = 0; j < d; j++)
				B(i, j) = B(i, j) / normB;
		}
	}

	Array<double > B;
	unsigned       d;
	double         cond;
};

#endif




