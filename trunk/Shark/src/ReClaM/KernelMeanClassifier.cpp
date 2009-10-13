//===========================================================================
/*!
 *  \file KernelMeanClassifier.cpp
 *
 *  \brief Kernel Mean Classifier
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


#include <ReClaM/KernelMeanClassifier.h>


KernelMeanClassifier::KernelMeanClassifier(KernelFunction* k)
: kernel(k)
{
	int p, pc = k->getParameterDimension();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) parameter(p) = k->getParameter(p);
}

KernelMeanClassifier::KernelMeanClassifier(const Array<double>& input, const Array<double>& target, KernelFunction* k)
: kernel(k)
{
	SetPoints(input, target);
}

KernelMeanClassifier::~KernelMeanClassifier()
{
}


void KernelMeanClassifier::SetPoints(const Array<double>& input, const Array<double>& target)
{
	inputDimension = input.dim(1);
	outputDimension = 1;
	training_input = input;
	training_target = target;
	bMustRecalc = true;
}

void KernelMeanClassifier::setParameter(unsigned int index, double value)
{
	parameter(index) = value;
	kernel->setParameter(index, value);
	bMustRecalc = true;
}

void KernelMeanClassifier::model(const Array<double>& input, Array<double>& output)
{
	if (bMustRecalc) Recalc();

	if (input.ndim() == 1)
	{
		output.resize(1, false);
		double v = bias;
		int i, l = training_target.dim(0);
		for (i = 0; i < l; i++)
		{
			if (training_target(i, 0) > 0.0) v += coeff_plus * kernel->eval(training_input[i], input);
			else v += coeff_minus * kernel->eval(training_input[i], input);
		}
		output(0) = v;
	}
	else if (input.ndim() == 2)
	{
		int j, s = input.dim(0);
		output.resize(s, 1, false);
		for (j = 0; j < s; j++)
		{
			double v = bias;
			int i, l = training_target.dim(0);
			for (i = 0; i < l; i++)
			{
				if (training_target(i, 0) > 0.0) v += coeff_plus * kernel->eval(training_input[i], input[j]);
				else v += coeff_minus * kernel->eval(training_input[i], input[j]);
			}
			output(j, 0) = v;
		}
	}
	else throw "[KernelMeanClassifier::model] invalid input dimension";
}

void KernelMeanClassifier::Recalc()
{
	bMustRecalc = false;

	int i, j, l = training_input.dim(0);
	int lp = 0;		// number of pos. examples
	int lm = 0;		// number of neg. examples

	// compute the class magnitudes and the resulting coefficients
		for (i = 0; i < l; i++) if (training_target(i, 0) > 0.0) lp++; else lm++;
	coeff_plus = 1.0 / lp;
	coeff_minus = -1.0 / lm;

	// compute the bias term
	double bp = 0.0;
	double bm = 0.0;
	for (i = 0; i < l; i++)
	{
		double y = training_target(i, 0);
		for (j = 0; j < l; j++)
		{
			if (training_target(j, 0) != y) continue;
			if (y > 0.0) bp += kernel->eval(training_input[i], training_input[j]);
			else bm += kernel->eval(training_input[i], training_input[j]);
		}
	}
	bias = 0.5 * (bm * coeff_minus * coeff_minus - bp * coeff_plus * coeff_plus);
}

