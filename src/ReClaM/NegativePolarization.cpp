/*!
*  \file NegativePolarization.cpp
*
*  \brief Implementation of the negative Kernel Polarization
*  	      Measure, that is, Kernel Target Alignment without normalization.
*
*  \author  T. Glasmachers
*
*  \par
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
*
*  <BR>
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


#include <ReClaM/NegativePolarization.h>


NegativePolarization::NegativePolarization()
{
}

NegativePolarization::~NegativePolarization()
{
}


double NegativePolarization::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// check the model type
	C_SVM* pCSVM = dynamic_cast<C_SVM*>(&model);
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	bool norm2 = false;
	if (pCSVM != NULL)
	{
		if (! pCSVM->is2norm()) throw SHARKEXCEPTION("[NegativePolarization::error] C_SVM must use 2-norm slack penalty.");
		pKernel = pCSVM->getSVM()->getKernel();
		norm2 = true;
	}
	else if (pKernel == NULL) throw SHARKEXCEPTION("[NegativePolarization::error] model is not a valid C_SVM or KernelFunction.");

	int i, j, T = input.dim(0);
	double k;
	double A = 0.0;
	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->eval(input[i], input[j]);
			A -= 2.0 * target(i, 0) * target(j, 0) * k;
		}
		k = pKernel->eval(input[i], input[i]);
		if (norm2) k -= (target(i, 0) > 0.0) ? 1.0 / pCSVM->get_Cplus() : 1.0 / pCSVM->get_Cminus();
		A -= k;
	}
	return A;
}

double NegativePolarization::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// check the model type
	C_SVM* pCSVM = dynamic_cast<C_SVM*>(&model);
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	bool norm2 = false;
	if (pCSVM != NULL)
	{
		if (! pCSVM->is2norm()) throw SHARKEXCEPTION("[NegativePolarization::errorDerivative] C_SVM must use 2-norm slack penalty.");
		pKernel = pCSVM->getSVM()->getKernel();
		norm2 = true;
	}
	else if (pKernel == NULL) throw SHARKEXCEPTION("[NegativePolarization::errorDerivative] model is not a valid C_SVM or KernelFunction.");

	int i, j, T = input.dim(0);
	int d, D = model.getParameterDimension();
	int dd = norm2 ? 1 : 0;
	double k;
	double yy;
	double A = 0.0;
	double invC;
	Array<double> der(D);
	derivative.resize(D + dd, false);
	derivative = 0.0;

	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->evalDerivative(input[i], input[j], der);
			yy = target(i, 0) * target(j, 0);
			for (d = 0; d < D; d++) derivative(d + dd) -= 2.0 * yy * der(d);
			A -= 2.0 * yy * k;
		}
		k = pKernel->evalDerivative(input[i], input[i], der);
		if (norm2)
		{
			if (target(i, 0) > 0.0)
			{
				invC = 1.0 / pCSVM->get_Cplus();
				k += invC;
				derivative(0) += invC * invC;
			}
			else
			{
				invC = 1.0 / pCSVM->get_Cminus();
				k += invC;
				derivative(0) += invC * invC;
			}
		}
		for (d = 0; d < D; d++) derivative(d + dd) -= der(d);
		A -= k;
	}

	return A;
}

