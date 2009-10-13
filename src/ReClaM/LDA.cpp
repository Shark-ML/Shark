//===========================================================================
/*!
 *  \file LDA.cpp
 *
 *  \brief Train a LinearClassifier using Linear Discriminant Analysis (LDA)
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


#include <ReClaM/LDA.h>
#include <LinAlg/LinAlg.h>


LDA::LDA()
{
}

LDA::~LDA()
{
}


void LDA::init(Model& model)
{
	LinearClassifier* lc = dynamic_cast<LinearClassifier*>(&model);
	AffineLinearFunction* alc = dynamic_cast<AffineLinearFunction*>(&model);
	if (lc == NULL && alc == NULL) throw SHARKEXCEPTION("[LDA::init] the model for LDA must be a LinearClassifier or an AffineLinearMap");
}

double LDA::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	LinearClassifier* lc = dynamic_cast<LinearClassifier*>(&model);
	AffineLinearFunction* alc = dynamic_cast<AffineLinearFunction*>(&model);
	if (lc != NULL) return optimize(*lc, input, target);
	else if (alc != NULL) return optimize(*alc, input, target);
	else throw SHARKEXCEPTION("[LDA::init] the model for LDA must be a LinearClassifier or an AffineLinearMap");
	return 0.0;
}

double LDA::optimize(LinearClassifier& model, const Array<double>& input, const Array<double>& target)
{
	// check dimensions
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	int i, ic = input.dim(0);
	int d, d2, dim = input.dim(1);
	SIZE_CHECK(input.dim(0) == target.dim(0));
	SIZE_CHECK((int)model.getInputDimension() == dim);
	SIZE_CHECK((int)target.dim(1) == model.getNumberOfClasses());

	// compute class means
	int j, c, cc = model.getNumberOfClasses();
	Array<int> num(cc);
	Array<double> mean(cc, dim);
	num = 0;
	mean = 0.0;
	for (i=0; i<ic; i++)
	{
		c = 0;
		for (j=0; j<cc; j++)
		{
			if (target(i, j) == 0.0) continue;
			if (target(i, j) != 1.0) throw SHARKEXCEPTION("[LDA::optimize] invalid class label");
			c = j;
			break;
		}
		num(c)++;
		for (d=0; d<dim; d++) mean(c, d) += input(i, d);
	}
	for (c=0; c<cc; c++)
	{
		if (num(c) == 0) throw SHARKEXCEPTION("[LDA::optimize] LDA can not handle a class without examples");
		for (d=0; d<dim; d++) mean(c, d) /= num(c);
	}

	// compute shared scatter matrix
	Array<double> diff(dim);
	Array<double> covariance(dim, dim);
	covariance = 0.0;
	for (i=0; i<ic; i++)
	{
		c = 0;
		for (j=0; j<cc; j++)
		{
			if (target(i, j) == 0.0) continue;
			if (target(i, j) != 1.0) throw SHARKEXCEPTION("[LDA::optimize] invalid class label");
			c = j;
			break;
		}
		for (d=0; d<dim; d++) diff(d) = input(i, d) - mean(c, d);
		for (d=0; d<dim; d++) for (d2=0; d2<dim; d2++) covariance(d, d2) += diff(d) * diff(d2);
	}

	// set parameters
	int p = 0;
	for (c=0; c<cc; c++)
	{
		for (d=0; d<dim; d++)
		{
			model.setParameter(p, mean(c, d));
			p++;
		}
	}
	for (d=0; d<dim; d++)
	{
		for (d2=0; d2<=d; d2++)
		{
			model.setParameter(p, covariance(d, d2));
			p++;
		}
	}

	return 0.0;
}

double LDA::optimize(AffineLinearFunction& model, const Array<double>& input, const Array<double>& target)
{
	// check dimensions
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);
	int i, ic = input.dim(0);
	int d, d2, dim = input.dim(1);
	SIZE_CHECK(input.dim(0) == target.dim(0));
	SIZE_CHECK((int)model.getInputDimension() == dim);
	SIZE_CHECK(target.dim(1) == 1);

	// compute class means
	int c;
	int num[2] = {0, 0};
	Array<double> mean(2, dim);
	mean = 0.0;
	for (i=0; i<ic; i++)
	{
		c = (int)target(i, 0);
		if (c == -1) c = 0;			// labels can be 0/1 or -1/+1
		if (c < 0 || c >= 2) throw SHARKEXCEPTION("[LDA::optimize] invalid class label");
		num[c]++;
		for (d=0; d<dim; d++) mean(c, d) += input(i, d);
	}
	for (c=0; c<2; c++)
	{
		if (num[c] == 0) throw SHARKEXCEPTION("[LDA::optimize] LDA can not handle a class without examples");
		for (d=0; d<dim; d++) mean(c, d) /= num[c];
	}

	// compute shared scatter matrix
	Array<double> diff(dim);
	Array2D<double> covariance(dim, dim);
	covariance = 0.0;
	for (i=0; i<ic; i++)
	{
		c = (int)target(i, 0);
		if (c == -1) c = 0;			// labels can be 0/1 or -1/+1
		for (d=0; d<dim; d++) diff(d) = input(i, d) - mean(c, d);
		for (d=0; d<dim; d++) for (d2=0; d2<dim; d2++) covariance(d, d2) += diff(d) * diff(d2);
	}

	// compute coefficients
	Array2D<double> inverse(dim, dim);
	invertSymm(inverse, covariance);
	for (d=0; d<dim; d++) diff(d) = 2.0 * (mean(1, d) - mean(0, d));
	Array<double> coeff(dim);
	matColVec(coeff, inverse, diff);
	double b = vecMatVec(mean[0], inverse, mean[0]) - vecMatVec(mean[1], inverse, mean[1]);

	// set parameters
	for (d=0; d<dim; d++) model.setParameter(d, coeff(d));
	model.setParameter(dim, b);

	return 0.0;
}

