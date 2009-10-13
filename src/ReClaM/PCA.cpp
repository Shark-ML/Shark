//===========================================================================
/*!
 *  \file PCA.cpp
 *
 *  \brief Train a (affine) linear map using Principal Component Analysis (PCA)
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


#include <LinAlg/LinAlg.h>
#include <ReClaM/PCA.h>


PCA::PCA()
{
	bWhitening = false;
}

PCA::~PCA()
{
}


void PCA::init(Model& model)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[PCA::init] the model for PCA must be an AffineLinearMap");
	if (alm->getInputDimension() < alm->getOutputDimension()) throw SHARKEXCEPTION("[PCA::init] PCA can not increase the dimensionality of the data");

	bWhitening = false;
}

void PCA::init(bool whitening)
{
	bWhitening = whitening;
}

double PCA::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[PCA::optimize] the model for PCA must be an AffineLinearMap");
	return optimize(*alm, input);
}

double PCA::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target, Array<double> &eigenvalues, Array<double> &trans)
{
	AffineLinearMap* alm = dynamic_cast<AffineLinearMap*>(&model);
	if (alm == NULL) throw SHARKEXCEPTION("[PCA::optimize] the model for PCA must be an AffineLinearMap");
	return optimize(*alm, input, eigenvalues, trans);
}

double PCA::optimize(AffineLinearMap& model, const Array<double>& input)
{
	Array<double> eigenvalues, trans;
	return optimize(model, input, eigenvalues, trans);
}


double PCA::optimize(AffineLinearMap& model, const Array<double>& input, Array<double> &eigenvalues, Array<double> &trans)
{
	int i, ic = model.getInputDimension();
	int o, oc = model.getOutputDimension();

	Array<double> mean;
	Array2D<double> scatter;
	MeanAndScatter(input, mean, scatter);

	Array2D<double> h(scatter);


	trans.resize(scatter, false);
	eigenvalues.resize(ic, false);

	if (bWhitening)
	{
		rankDecomp(scatter, trans, h, eigenvalues);
	}
	else
	{
		eigensymm(scatter, trans, eigenvalues);
	}

	// set parameters
	int p = 0;
	for (o=0; o<oc; o++)
	{
		for (i=0; i<ic; i++)
		{
			model.setParameter(p, trans(i, o));
			p++;
		}
	}
	for (o=0; o<oc; o++)
	{
		double value = 0.0;
		for (i=0; i<ic; i++) value -= trans(i, o) * mean(i);
		model.setParameter(p, value);
		p++;
	}

	return 0.0;
}

void PCA::MeanAndScatter(const Array<double>& input, Array<double>& mean, Array2D<double>& scatter)
{
	SIZE_CHECK(input.ndim() == 2);
	int i, ic = input.dim(0);
	int d, d2, dim = input.dim(1);

	// compute the mean
	mean.resize(dim, false);
	mean = 0.0;
	for (i=0; i<ic; i++)
	{
		for (d=0; d<dim; d++) mean(d) += input(i, d);
	}
	for (d=0; d<dim; d++) mean(d) /= ic;

	// compute scatter matrix
	Array<double> diff(dim);
	scatter.resize(dim, dim, false);
	scatter = 0.0;
	for (i=0; i<ic; i++)
	{
		for (d=0; d<dim; d++) diff(d) = input(i, d) - mean(d);
		for (d=0; d<dim; d++) for (d2=0; d2<dim; d2++) scatter(d, d2) += diff(d) * diff(d2);
	}
	for (d=0; d<dim; d++) for (d2=0; d2<dim; d2++) scatter(d, d2) /= ic;
}

