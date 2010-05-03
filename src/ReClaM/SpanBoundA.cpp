//===========================================================================
/*!
*  \file SpanBoundA.cpp
*
*  \brief Compute the approximate Span-Bound for the 1-norm SVM
*
*  \author  T. Glasmachers
*  \date    2010
*
*
*  \par Copyright (c) 1999-2010:
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


#include <ReClaM/SpanBoundA.h>
#include <ReClaM/Svm.h>
#include <LinAlg/VecMat.h>


SpanBoundA::SpanBoundA()
{
	maxIter = -1;
}

SpanBoundA::~SpanBoundA()
{
}


double SpanBoundA::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL) throw SHARKEXCEPTION("[SpanBoundA::error] model is not a valid C_SVM");

	bool norm2 = csvm->is2norm();
	SVM* svm = csvm->getSVM();
	if (norm2) throw SHARKEXCEPTION("[SpanBoundA::error] model is not a valid 1-norm C_SVM");
	double C_plus = csvm->get_Cplus();
	double C_minus = csvm->get_Cminus();
	KernelFunction& kernel = *svm->getKernel();

	SVM_Optimizer svmopt;
	svmopt.init(*csvm);
	svmopt.setMaxIterations(maxIter);
	svmopt.optimize(*svm, input, target);

	QpSvmDecomp* solver = (QpSvmDecomp*)svmopt.get_Solver();
	if (! solver->isOptimal()) return 1e100;
	iter = solver->iterations();

	unsigned int i, examples = input.dim(0);
	unsigned int sv = 0, bsv = 0;
	for (i=0; i<examples; i++)
	{
		double a = svm->getParameter(i);
		if (a != 0.0)
		{
			if (a == -C_minus || a == C_plus) bsv++;
			else sv++;
		}
	}
	std::vector<unsigned int> map(sv);
	std::vector<double> alpha(sv);
	unsigned int j = 0;
	for (i=0; i<examples; i++)
	{
		double a = svm->getParameter(i);
		if (a == 0.0) continue;
		if (a == C_plus || a == -C_minus) continue;

		map[j] = i;
		alpha[j] = fabs(a);
		j++;
	}

	// compute the squared span S2
	Array<double> S2(sv);
	Matrix tildeK(sv+1, sv+1);
	for (i=0; i<sv; i++)
	{
		for (j=0; j<i; j++)
		{
			double k = kernel.eval(input[map[i]], input[map[j]]);
			tildeK(i, j) = tildeK(j, i) = k;
		}
		double k = kernel.eval(input[map[i]], input[map[i]]);
		tildeK(i, i) = k;
		tildeK(i, sv) = tildeK(sv, i) = 1.0;
	}
	tildeK(sv, sv) = 0.0;

	Matrix tildeKinv = tildeK.inverseSymm();
	for (i=0; i<sv; i++) S2(i) = 1.0 / tildeKinv(i, i);

	// return the span bound value
	double sum = bsv;
	for (i=0; i<sv; i++) sum += alpha[i] * S2(i);
	return (double)sum / (double)examples;
}

double SpanBoundA::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL) throw SHARKEXCEPTION("[SpanBoundA::error] model is not a valid C_SVM");

	bool norm2 = csvm->is2norm();
	SVM* svm = csvm->getSVM();
	if (norm2) throw SHARKEXCEPTION("[SpanBoundA::error] model is not a valid 1-norm C_SVM");
	double C_plus = csvm->get_Cplus();
	double C_minus = csvm->get_Cminus();
	KernelFunction& kernel = *svm->getKernel();
	unsigned int p, pc = kernel.getParameterDimension();

	SVM_Optimizer svmopt;
	svmopt.init(*csvm);
	svmopt.setMaxIterations(maxIter);
	svmopt.optimize(*svm, input, target);

	QpSvmDecomp* solver = (QpSvmDecomp*)svmopt.get_Solver();
	if (! solver->isOptimal()) return 1e100;
	iter = solver->iterations();

	unsigned int i, examples = input.dim(0);
	unsigned int sv = 0, bsv = 0;
	for (i=0; i<examples; i++)
	{
		double a = svm->getParameter(i);
		if (a != 0.0)
		{
			if (a == -C_minus || a == C_plus) bsv++;
			else sv++;
		}
	}
	std::vector<unsigned int> map(sv);
	std::vector<double> alpha(sv);
	unsigned int j = 0;
	for (i=0; i<examples; i++)
	{
		double a = svm->getParameter(i);
		if (a == 0.0) continue;
		if (a == C_plus || a == -C_minus) continue;

		map[j] = i;
		alpha[j] = fabs(a);
		j++;
	}

	// compute the squared span S2
	Array<double> S2(sv);
	Array<double> S2der(sv, pc);
	Matrix tildeK(sv+1, sv+1);
	Array<double> tildeKder(sv, sv, pc);
	for (i=0; i<sv; i++)
	{
		for (j=0; j<i; j++)
		{
			Array<double> der(pc);
			double k = kernel.evalDerivative(input[map[i]], input[map[j]], der);
			tildeKder[i][j] = der;
			tildeKder[j][i] = der;
			tildeK(i, j) = tildeK(j, i) = k;
		}
		double k = kernel.eval(input[map[i]], input[map[i]]);
		tildeK(i, i) = k;
		tildeK(i, sv) = tildeK(sv, i) = 1.0;
	}
	tildeK(sv, sv) = 0.0;

	Matrix tildeKinv = tildeK.inverseSymm();
	for (i=0; i<sv; i++) S2(i) = 1.0 / tildeKinv(i, i);

	// compute the derivatives of the squared span
	// w.r.t. all kernel parameters
	Matrix D(sv+1, sv+1);
	for (i=0; i<sv+1; i++) D(sv, i) = D(i, sv) = 0.0;
	for (p=0; p<pc; p++)
	{
		for (i=0; i<sv; i++) for (j=0; j<sv; j++) D(i, j) = tildeKder(i, j, p);
		Matrix tmp = tildeKinv * D * tildeKinv;
		for (i=0; i<sv; i++) S2der(i, p) = S2(i) * S2(i) * tmp(i, i);
	}

	// compute the derivative of the alpha parameters
	// w.r.t. the kernel parameters and C
	const Array<double>& alphaDer = csvm->PrepareDerivative();

	// compose the derivative
	derivative.resize(pc + 1, false);
	for (p=0; p<pc; p++)
	{
		double sum = 0.0;
		for (i=0; i<sv; i++) sum += alpha[i] * S2der(i, p) + alphaDer(map[i], p + 1) * S2(i);
		derivative(p + 1) = sum;
	}
	double sum = 0.0;
	for (i=0; i<sv; i++) sum += alphaDer(map[i], 0) * S2(i);
	derivative(0) = sum;
	derivative /= (double)examples;

	// return the span bound value
	sum = bsv;
	for (i=0; i<sv; i++) sum += alpha[i] * S2(i);
	return (double)sum / (double)examples;
}
