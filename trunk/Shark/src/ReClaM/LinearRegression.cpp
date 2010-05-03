//===========================================================================
/*!
 *  \file LinearRegression.cpp
 *
 *  \brief Linear Regression
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par
 *      This implementation is based upon a class removed from
 *      the LinAlg package, written by M. Kreutz in 1998.
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
 *  <BR><HR>1
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


#include <LinAlg/LinAlg.h>
#include <ReClaM/LinearRegression.h>


LinearRegression::LinearRegression()
{
}

LinearRegression::~LinearRegression()
{
}


void LinearRegression::init(Model& model)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[LinearRegression::init] the model for Linear Regression must be an AffineLinearMap");
}

double LinearRegression::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[LinearRegression::optimize] the model for Linear Regression must be an AffineLinearMap");
	return optimize(*alm, input, target);
}

double LinearRegression::optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& target)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);

	int i, j, ic = model.getInputDimension();
	int o, oc = model.getOutputDimension();
	int e, ec = input.dim(0);

	SIZE_CHECK((int)input.dim(1) == ic);
	SIZE_CHECK((int)target.dim(1) == oc);
	SIZE_CHECK((int)target.dim(0) == ec);

	double value;
	for (o=0; o<oc; o++)
	{
		Array2D<double> X(ec, ic+1);
		for (e=0; e<ec; e++)
		{
			for (i=0; i<ic; i++) X(e, i) = input(e, i);
			X(e, ic) = 1.0;
		}

		Array2D<double> XTX(ic+1, ic+1);
		for (i=0; i<=ic; i++)
		{
			for (j=0; j<i; j++)
			{
				value = 0.0;
				for (e=0; e<ec; e++) value += X(e, i) * X(e, j);
				XTX(i, j) = XTX(j, i) = value;
			}
			value = 0.0;
			for (e=0; e<ec; e++) value += X(e, i) * X(e, i);
			XTX(i, i) = value;
		}

		Array2D<double> XTXinv(ic+1, ic+1);
		invertSymm(XTXinv, XTX);

		Array<double> XTy(ic+1);
		for (i=0; i<=ic; i++)
		{
			value = 0.0;
			for (e=0; e<ec; e++) value += X(e, i) * target(e);
			XTy(i) = value;
		}

		Array<double> beta(ic+1);
		for (i=0; i<=ic; i++)
		{
			value = 0.0;
			for (j=0; j<=ic; j++) value += XTXinv(i, j) * XTy(j);
			beta(i) = value;
		}

		int p = ic * o;
		for (i=0; i<ic; i++)
		{
			model.setParameter(p, beta(i));
			p++;
		}
		model.setParameter(ic * oc + o, beta(ic));
	}

	return 0.0;
}

