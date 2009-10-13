//===========================================================================
/*!
*  \file ValidationError.cpp
*
*  \brief Compute the error on a hold out set
*
*  \author  T. Glasmachers
*  \date    2008
*
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


#include <vector>
#include <Rng/GlobalRng.h>
#include <ReClaM/ValidationError.h>


ValidationError::ValidationError(ErrorFunction* base, Optimizer* opt, int iter, double holdOutFraction)
{
	this->baseError = base;
	this->optimizer = opt;
	this->iterations = iter;
	this->holdOut = holdOutFraction;
}

ValidationError::~ValidationError()
{
}


double ValidationError::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// split the data into training and validation set
	int i, ic = input.dim(0);
	int dim = input.dim(1);
	int tdim = target.dim(1);
	int train = (int)((1.0 - holdOut) * ic);
	int validation = ic - train;
	std::vector<int> tr(ic);
	std::vector<int> val(validation);
	for (i=0; i<ic; i++) tr[i] = i;
	for (i=0; i<validation; i++)
	{
		int n = Rng::discrete(0, tr.size() - 1);
		val[i] = tr[n];
		tr.erase(tr.begin() + n);
	}
	Array<double> trainD(train, dim);
	Array<double> trainT(train, tdim);
	Array<double> validationD(validation, dim);
	Array<double> validationT(validation, tdim);
	for (i=0; i<train; i++)
	{
		trainD[i] = input[tr[i]];
		trainT[i] = target[tr[i]];
	}
	for (i=0; i<validation; i++)
	{
		validationD[i] = input[val[i]];
		validationT[i] = target[val[i]];
	}

	// train the model on the training subset
	optimizer->init(model);
	for (i=0; i<iterations; i++) optimizer->optimize(model, *baseError, trainD, trainT);

	// return the validation error
	return baseError->error(model, validationD, validationT);
}

double ValidationError::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// split the data into training and validation set
	int i, ic = input.dim(0);
	int dim = input.dim(1);
	int tdim = target.dim(1);
	int train = (int)(holdOut * ic);
	int validation = ic - train;
	std::vector<int> tr(ic);
	std::vector<int> val(validation);
	for (i=0; i<ic; i++) tr[i] = i;
	for (i=0; i<validation; i++)
	{
		int n = Rng::discrete(0, tr.size() - 1);
		val[i] = tr[n];
		tr.erase(tr.begin() + n);
	}
	Array<double> trainD(train, dim);
	Array<double> trainT(train, tdim);
	Array<double> validationD(validation, dim);
	Array<double> validationT(validation, tdim);
	for (i=0; i<train; i++)
	{
		trainD[i] = input[tr[i]];
		trainT[i] = target[tr[i]];
	}
	for (i=0; i<validation; i++)
	{
		validationD[i] = input[val[i]];
		validationT[i] = target[val[i]];
	}

	// train the model on the training subset
	optimizer->init(model);
	for (i=0; i<iterations; i++) optimizer->optimize(model, *baseError, trainD, trainT);

	// return the validation error
	return baseError->errorDerivative(model, validationD, validationT, derivative);
}
