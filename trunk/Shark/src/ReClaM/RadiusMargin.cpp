//===========================================================================
/*!
 *  \file RadiusMargin.cpp
 *
 *  \brief Squared radius margin quotient
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


#include <ReClaM/RadiusMargin.h>


RadiusMargin::RadiusMargin()
{
	maxIter = -1;
}

RadiusMargin::~RadiusMargin()
{
}


double RadiusMargin::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	// check the model type
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL)
		throw SHARKEXCEPTION("[RadiusMargin::error] model is not a valid C_SVM.");

	double Cplus = csvm->get_Cplus();
	double Cminus = csvm->get_Cminus();
	bool norm2 = csvm->is2norm();
	SVM* pSVM = csvm->getSVM();

	// compute the coefficients alpha and beta
	Array<double> alpha;
	Array<double> beta;
	unsigned int i, examples = input.dim(0);
	double R2 = solveProblems(pSVM, Cplus, Cminus, input, target, alpha, beta, norm2);
	if (R2 < 0.0) return 1e100;

	double w2 = 0.0;
	for (i = 0; i < examples; i++) w2 += alpha(i);

	return R2 * w2;
}

double RadiusMargin::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	// check the model type
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL)
		throw SHARKEXCEPTION("[RadiusMargin::errorDerivative] model is not a valid C_SVM.");

	double Cplus = csvm->get_Cplus();
	double Cminus = csvm->get_Cminus();
	bool norm2 = csvm->is2norm();
	SVM* pSVM = csvm->getSVM();

	// compute the coefficients alpha and beta
	Array<double> alpha;
	Array<double> beta;
	KernelFunction* kernel = pSVM->getKernel();
	unsigned int k, kc = kernel->getParameterDimension();
	derivative.resize(kc + 1, false);
	double R2 = solveProblems(pSVM, Cplus, Cminus, input, target, alpha, beta, norm2);
	if (R2 < 0.0)
	{
		derivative = 0.0;
		return 1e100;
	}

	unsigned int i, j, examples = input.dim(0);
	double w2;
	double kv;
	double ayi, ayj, bi, bj;
	Array<double> dw2(kc);
	Array<double> dR2(kc);
	Array<double> dkv(kc);

	w2 = 0.0;
	dw2 = 0.0;
	dR2 = 0.0;
	double dR2dCplus = 0.0;
	double dR2dCminus = 0.0;
	double dw2dCplus = 0.0;
	double dw2dCminus = 0.0;
	for (j = 0; j < examples; j++)
	{
		ayj = alpha(j) * target(j, 0);
		bj = beta(j);
		for (i = 0; i < examples; i++)
		{
			ayi = alpha(i) * target(i, 0);
			bi = beta(i);
			kv = kernel->evalDerivative(input[i], input[j], dkv);
			if (i == j)
			{
				if (norm2)
				{
					double invC;
					if (target(i, 0) > 0.0)
					{
						invC = 1.0 / Cplus;
						kv += invC;
						dR2dCplus -= bi * invC * invC;
						dw2dCplus += (ayi * ayi) * invC * invC;
					}
					else
					{
						invC = 1.0 / Cminus;
						kv += invC;
						dR2dCminus -= (bi - bi * bi) * invC * invC;
						dw2dCminus += (ayi * ayi) * invC * invC;
					}
				}
				w2 += alpha(i);
				for (k = 0; k < kc; k++) dR2(k) += bi * dkv(k);
			}
			for (k = 0; k < kc; k++)
			{
				dw2(k) -= ayi * ayj * dkv(k);
				dR2(k) -= bi * bj * dkv(k);
			}
		}
	}

	if (norm2)
	{
		double cr = csvm->getCRatio();
		derivative(0) = R2 * (dw2dCplus + cr * dw2dCminus) + w2 * (dR2dCplus + cr * dR2dCminus);
	}
	else
	{
		derivative(0) = 0.0;
	}
	for (k = 0; k < kc; k++) derivative(k + 1) = R2 * dw2(k) + w2 * dR2(k);

	return R2 * w2;
}

double RadiusMargin::solveProblems(SVM* pSVM, double Cplus, double Cminus, const Array<double>& input, const Array<double>& target, Array<double>& alpha, Array<double>& beta, bool norm2)
{
	unsigned int e, examples = input.dim(0);
	KernelFunction* kernel = pSVM->getKernel();

	alpha.resize(examples, false);
	beta.resize(examples, false);
	alpha = 0.0;
	beta = 1.0 / examples;

	Array<double> svm_linear(examples);
	Array<double> radius_linear(examples);
	Array<double> diag(examples);
	Array<double> lower(examples);
	Array<double> upper(examples);
	Array<double> zero(examples);
	Array<double> inf(examples);
	if (norm2)
	{
		for (e = 0; e < examples; e++)
		{
			svm_linear(e) = target(e, 0);
			if (target(e, 0) > 0.0)
			{
				radius_linear(e) = 0.5 * (kernel->eval(input[e], input[e]) + 1.0 / Cplus);
				diag(e) = 1.0 / Cplus;
				lower(e) = 0.0;
				upper(e) = 1e100;
			}
			else
			{
				radius_linear(e) = 0.5 * (kernel->eval(input[e], input[e]) + 1.0 / Cminus);
				diag(e) = 1.0 / Cminus;
				lower(e) = -1e100;
				upper(e) = 0.0;
			}
		}
	}
	else
	{
		for (e = 0; e < examples; e++)
		{
			svm_linear(e) = target(e, 0);
			radius_linear(e) = 0.5 * (kernel->eval(input[e], input[e]));
			diag(e) = 0.0;
			if (target(e, 0) > 0.0)
			{
				lower(e) = 0.0;
				upper(e) = Cplus;
			}
			else
			{
				lower(e) = -Cminus;
				upper(e) = 0.0;
			}
		}
	}
	for (e = 0; e < examples; e++)
	{
		zero(e) = 0.0;
		inf(e) = 1e100;
	}

	RegularizedKernelMatrix* km = new RegularizedKernelMatrix(kernel, input, diag);
	CachedMatrix cm(km);
	QpSvmDecomp qp(cm);
	qp.setMaxIterations(maxIter);

	// solve the first problem to obtain the alpha coefficients
	qp.Solve(svm_linear, lower, upper, alpha);
	if (! qp.isOptimal()) return -1.0;
	iter = qp.iterations();

	// solve the second problem to obtain the beta coefficients
	double R2 = 2.0 * qp.Solve(radius_linear, zero, inf, beta);
	if (! qp.isOptimal()) return -1.0;
	if (qp.iterations() > iter) iter = qp.iterations();

	// the alpha's are assumed to be positive in the following
	for (e = 0; e < examples; e++) alpha(e) = fabs(alpha(e));

	return R2;
}

