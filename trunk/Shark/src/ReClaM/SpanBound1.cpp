//===========================================================================
/*!
*  \file SpanBound1.cpp
*
*  \brief Compute the SpanBound for the 1-norm SVM
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


//
// This is a very brutal and probably inefficient implementation.
// At least, it works, and the derivatives seem to be correct, too.
//


#include <ReClaM/SpanBound1.h>
#include <ReClaM/QuadraticProgram.h>


SpanBound1::SpanBound1(bool verbose)
{
	this->verbose = verbose;
	this->maxiter = -1;
}

SpanBound1::~SpanBound1()
{
}


double SpanBound1::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL || csvm->is2norm()) throw SHARKEXCEPTION("[SpanBound1::error] model is not a valid 1-norm C_SVM");

	SVM* svm = csvm->getSVM();
	double Cplus = csvm->get_Cplus();
	double Cminus = csvm->get_Cminus();

	// first compute the SVM hypothesis
	SVM_Optimizer svmopt;
	svmopt.init(*csvm);
	svmopt.setMaxIterations(maxiter);
	svmopt.optimize(*svm, input, target);
	QpSvmDecomp* solver = (QpSvmDecomp*)svmopt.get_Solver();
	if (! solver->isOptimal()) return 1e100;

	double ret = 0.0;

	// for all support vectors
	int i, ic = svm->getExamples();
	for (i=0; i<ic; i++)
	{
		double alpha = svm->getAlpha(i);
		if (alpha != 0.0)
		{
			// compute f^{\hat p}(x_p)
			double hhat = target(i) * bound(csvm, i, input, target);
			if (hhat <= 0.0)
			{
				ret += 1.0;
				continue;
			}

			if (alpha == Cplus || alpha == -Cminus)
			{
				// bounded SV
				double hx = target(i) * svm->model(input[i]);
				ret += (hx - hhat) / hx;
			}
			else
			{
				// free SV
				ret += 1.0 - hhat;
			}
		}
	}

	return ret;
}

double SpanBound1::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL || csvm->is2norm()) throw SHARKEXCEPTION("[SpanBound1::errorDerivative] model is not a valid 1-norm C_SVM");

	int p, pc = csvm->getParameterDimension();
	derivative.resize(pc, false);
	derivative = 0.0;

	SVM* svm = csvm->getSVM();
	double Cplus = csvm->get_Cplus();
	double Cminus = csvm->get_Cminus();

	// first compute the SVM hypothesis
	SVM_Optimizer svmopt;
	svmopt.init(*csvm);
	svmopt.setMaxIterations(maxiter);
	svmopt.optimize(*svm, input, target);
	QpSvmDecomp* solver = (QpSvmDecomp*)svmopt.get_Solver();
	if (! solver->isOptimal()) return 1e100;

	csvm->PrepareDerivative();

	double ret = 0.0;

	// for all support vectors
	int i, ic = svm->getExamples();
	for (i=0; i<ic; i++)
	{
		double alpha = svm->getAlpha(i);
		double y = target(i, 0);
		if (alpha != 0.0)
		{
			// compute f^{\hat p}(x_p)
			Array<double> der_hhat;
			double hhat = boundDerivative(csvm, i, input, target, der_hhat);
			if (y * hhat <= 0.0)
			{
				ret += 1.0;
				continue;
			}

			if (alpha == Cplus || alpha == -Cminus)
			{
				// bounded SV
				double h = svm->model(input[i]);
				Array<double> der_h;
				csvm->modelDerivative(input[i], der_h);

				ret += (h - hhat) / h;
				for (p=0; p<pc; p++) derivative(p) += (der_h(0, p) * hhat - der_hhat(0, p) * h) / (h * h);
			}
			else
			{
				// free SV
				ret += 1.0 - y * hhat;
				for (p=0; p<pc; p++) derivative(p) -= y * der_hhat(0, p);
			}
		}
	}

	return ret;
}

double SpanBound1::bound(C_SVM* csvm, int p, const Array<double>& input, const Array<double>& target)
{
	int i;
	int ell = input.dim(0);
	int dim = input.dim(1);
	Array<double> _input(ell - 1, dim);
	Array<double> _target(ell - 1, 1);

	for (i=0; i<ell; i++)
	{
		if (i < p)
		{
			_input[i] = input[i];
			_target[i] = target[i];
		}
		else if (i > p)
		{
			_input[i - 1] = input[i];
			_target[i - 1] = target[i];
		}
	}

	SVM svm(csvm->getSVM()->getKernel());
	C_SVM csvm2(&svm, csvm->get_Cplus(), csvm->get_Cminus());

	SVM_Optimizer svmopt;
	svmopt.init(csvm2);
	svmopt.optimize(svm, _input, _target);

	return svm.model(input[p]);
}

double SpanBound1::boundDerivative(C_SVM* csvm, int p, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	int i;
	int ell = input.dim(0);
	int dim = input.dim(1);
	Array<double> _input(ell - 1, dim);
	Array<double> _target(ell - 1, 1);

	for (i=0; i<p; i++)
	{
		_input[i] = input[i];
		_target[i] = target[i];
	}
	for (i=p+1; i<ell; i++)
	{
		_input[i - 1] = input[i];
		_target[i - 1] = target[i];
	}

	SVM svm(csvm->getSVM()->getKernel());
	C_SVM csvm2(&svm, csvm->get_Cplus(), csvm->get_Cminus(), csvm->is2norm(), csvm->isUnconstrained());

	SVM_Optimizer svmopt;
	svmopt.init(csvm2);
	svmopt.optimize(svm, _input, _target);
	double ret = svm.model(input[p]);

	// compute the derivative only if it is really used
	if (ret * target(p, 0) <= 0.0) return ret;

	csvm2.PrepareDerivative();
	csvm2.modelDerivative(input[p], derivative);
	return ret;
}
