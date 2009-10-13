/*!
 *  \file InverseClassSeparability.cpp
 *
 *  \brief Inverse of the Class Separability Measure J by Huilin Xiong and M. N. S. Swamy
 *
 *  \author  T. Glasmachers
 *  \date    2006
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


#include <ReClaM/InverseClassSeparability.h>


InverseClassSeparability::InverseClassSeparability()
{
}

InverseClassSeparability::~InverseClassSeparability()
{
}


double InverseClassSeparability::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// check the model type
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	if (pKernel == NULL) throw SHARKEXCEPTION("[InverseClassSeparability::error] model is not a valid KernelFunction.");

	int lPlus = 0;
	int lMinus = 0;
	int i, j, t, T = input.dim(0);
		for (t = 0; t < T; t++) if (target(t, 0) > 0.0) lPlus++; else lMinus++;
	int l = lPlus + lMinus;
	double lPlusInverse = 1.0 / lPlus;
	double lMinusInverse = 1.0 / lMinus;
	double lInverse = 1.0 / l;

	double B = 0.0;
	double W = 0.0;
	double k;
	double b, w;
	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->eval(input[i], input[j]);
			if (target(i, 0) > 0.0)
			{
				if (target(j, 0) > 0.0)
				{
					b = lPlusInverse - lInverse;
					w = -lPlusInverse;
				}
				else
				{
					b = -lInverse;
					w = 0.0;
				}
			}
			else
			{
				if (target(j, 0) > 0.0)
				{
					b = -lInverse;
					w = 0.0;
				}
				else
				{
					b = lMinusInverse - lInverse;
					w = -lMinusInverse;
				}
			}
			B += 2.0 * b * k;
			W += 2.0 * w * k;
		}

		k = pKernel->eval(input[i], input[i]);
		if (target(i, 0) > 0.0)
		{
			b = lPlusInverse - lInverse;
			w = 1.0 - lPlusInverse;
		}
		else
		{
			b = lMinusInverse - lInverse;
			w = 1.0 - lMinusInverse;
		}
		B += b * k;
		W += w * k;
	}

	return W / B;
}

double InverseClassSeparability::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// check the model type
	KernelFunction* pKernel = dynamic_cast<KernelFunction*>(&model);
	if (pKernel == NULL) throw SHARKEXCEPTION("[InverseClassSeparability::errorDerivative] model is not a valid KernelFunction.");

	int p, pc = pKernel->getParameterDimension();
	derivative.resize(pc, false);
	Array<double> der(pc);

	int lPlus = 0;
	int lMinus = 0;
	int i, j, t, T = input.dim(0);
		for (t = 0; t < T; t++) if (target(t, 0) > 0.0) lPlus++; else lMinus++;
	int l = lPlus + lMinus;
	double lPlusInverse = 1.0 / lPlus;
	double lMinusInverse = 1.0 / lMinus;
	double lInverse = 1.0 / l;

	double B = 0.0;
	double W = 0.0;
	Array<double> gB(pc);
	Array<double> gW(pc);
	gB = 0.0;
	gW = 0.0;
	double k;
	double b, w;
	for (i = 0; i < T; i++)
	{
		for (j = 0; j < i; j++)
		{
			k = pKernel->evalDerivative(input[i], input[j], der);
			if (target(i, 0) > 0.0)
			{
				if (target(j, 0) > 0.0)
				{
					b = lPlusInverse - lInverse;
					w = -lPlusInverse;
				}
				else
				{
					b = -lInverse;
					w = 0.0;
				}
			}
			else
			{
				if (target(j, 0) > 0.0)
				{
					b = -lInverse;
					w = 0.0;
				}
				else
				{
					b = lMinusInverse - lInverse;
					w = -lMinusInverse;
				}
			}
			B += 2.0 * b * k;
			W += 2.0 * w * k;
			for (p = 0; p < pc; p++)
			{
				gB(p) += 2.0 * b * der(p);
				gW(p) += 2.0 * w * der(p);
			}
		}

		k = pKernel->evalDerivative(input[i], input[i], der);
		if (target(i, 0) > 0.0)
		{
			b = lPlusInverse - lInverse;
			w = 1.0 - lPlusInverse;
		}
		else
		{
			b = lMinusInverse - lInverse;
			w = 1.0 - lMinusInverse;
		}
		B += b * k;
		W += w * k;
		for (p = 0; p < pc; p++)
		{
			gB(p) += b * der(p);
			gW(p) += w * der(p);
		}
	}

	double ret = W / B;
	for (p = 0; p < pc; p++) derivative(p) = (gW(p) - ret * gB(p)) / B;
	return ret;
}

