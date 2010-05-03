//===========================================================================
/*!
 *  \file LOO.cpp
 *
 *  \brief Leave One Out (LOO) Error for Support Vector Machines
 *
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
 *   \par Project:
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


#include <ReClaM/LOO.h>
#include <ReClaM/Svm.h>


LOO::LOO()
{
	maxIter = -1;
}

LOO::~LOO()
{
}


double LOO::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL) throw SHARKEXCEPTION("[LOO::error] model is not a valid C_SVM");

	double Cplus = csvm->get_Cplus();
	double Cminus = csvm->get_Cminus();
	bool norm2 = csvm->is2norm();
	SVM* svm = csvm->getSVM();
	KernelFunction* kernel = svm->getKernel();

	svm->SetTrainingData(input);

	int i, j, examples = input.dim(0);
	Array<double> diagMod(examples);
	Array<double> linear(examples);
	Array<double> lower(examples);
	Array<double> upper(examples);
	Array<double> alpha(examples);
	double b;

	if (norm2)
	{
		for (i = 0; i < examples; i++)
		{
			alpha(i) = 0.0;
			linear(i) = target(i, 0);
			if (target(i, 0) > 0.0)
			{
				diagMod(i) = 1.0 / Cplus;
				lower(i) = 0.0;
				upper(i) = 1e100;
			}
			else
			{
				diagMod(i) = 1.0 / Cminus;
				lower(i) = -1e100;
				upper(i) = 0.0;
			}
		}
	}
	else
	{
		for (i = 0; i < examples; i++)
		{
			alpha(i) = 0.0;
			linear(i) = target(i, 0);
			diagMod(i) = 0.0;
			if (target(i, 0) > 0.0)
			{
				lower(i) = 0.0;
				upper(i) = Cplus;
			}
			else
			{
				lower(i) = -Cminus;
				upper(i) = 0.0;
			}
		}
	}

	// train the full SVM and remember the solution
	RegularizedKernelMatrix* km = new RegularizedKernelMatrix(kernel, input, diagMod);
	CachedMatrix cm(km);
	QpSvmDecomp qp(cm);
	qp.setMaxIterations(maxIter);
	qp.Solve(linear, lower, upper, alpha);
// 	if (! qp.isOptimal()) return 1e100;
	iter = qp.iterations();
	b = ComputeB(qp, lower, upper, alpha);

	// leave one out
	int mistakes = 0;
	Array<double> loo_alpha;
	double loo_lower, loo_upper;
	for (i = 0; i < examples; i++)
	{
		if (alpha(i) == 0.0) continue;

		loo_alpha = alpha;

		// remove the i-th example and construct
		// a closeby feasible point
		double diff = -loo_alpha(i);
		loo_alpha(i) = 0.0;
		loo_lower = lower(i);
		loo_upper = upper(i);
		lower(i) = 0.0;
		upper(i) = 0.0;
		for (j = 0; j < examples; j++)
		{
			if (j == i) continue;

			if (diff > 0.0 && loo_alpha(j) > 0.0)
			{
				if (diff > loo_alpha(j))
				{
					diff -= loo_alpha(j);
					loo_alpha(j) = 0.0;
				}
				else
				{
					loo_alpha(j) -= diff;
					diff = 0.0;
					break;
				}
			}
			else if (diff < 0.0 && loo_alpha(j) < 0.0)
			{
				if (diff < loo_alpha(j))
				{
					diff -= loo_alpha(j);
					loo_alpha(j) = 0.0;
				}
				else
				{
					loo_alpha(j) -= diff;
					diff = 0.0;
					break;
				}
			}
		}

		// compute the solution
		qp.Solve(linear, lower, upper, loo_alpha);
// 		if (! qp.isOptimal()) return 1e100;
		b = ComputeB(qp, lower, upper, loo_alpha);

		// check for an error
		if ((qp.ComputeInnerProduct(i, loo_alpha) + b) * target(i, 0) <= 0.0)
			mistakes++;

		lower(i) = loo_lower;
		upper(i) = loo_upper;
	}

	// return the error
	return (double)mistakes / (double)examples;
}

double LOO::ComputeB(QpSvmDecomp& qp, const Array<double>& lower, const Array<double>& upper, const Array<double>& alpha)
{
	// computation of b
	Array<double> gradient;
	qp.getGradient(gradient);
	double lowerBound = -1e100;
	double upperBound = 1e100;
	double sum = 0.0;
	unsigned int freeVars = 0;
	double value;
	int e, examples = gradient.dim(0);
	for (e = 0; e < examples; e++)
	{
		value = gradient(e);
		if (alpha(e) == lower(e))
		{
			if (value > lowerBound) lowerBound = value;
		}
		else if (alpha(e) == upper(e))
		{
			if (value < upperBound) upperBound = value;
		}
		else
		{
			sum += value;
			freeVars++;
		}
	}

	if (freeVars > 0)
		return sum / freeVars;						// stabilized exact value
	else
		return 0.5 *(lowerBound + upperBound);		// best estimate
}

void LOO::setMaxIterations(SharkInt64 maxiter)
{
	this->maxIter = maxiter;
}
