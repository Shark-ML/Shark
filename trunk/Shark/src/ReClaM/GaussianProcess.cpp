//===========================================================================
/*!
 *  \file GaussianProcess.cpp
 *
 *  \brief Gaussian Process implementation
 *
 *  \author  C. Igel
 *  \date    2007
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


#include <math.h>
#include <SharkDefs.h>
#include <LinAlg/LinAlg.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/GaussianProcess.h>


GaussianProcess::GaussianProcess(SVM* svm, double precision)
: MetaSVM(svm, 1)
{
	setBetaInv(precision);
}

GaussianProcess::GaussianProcess(SVM* svm, const Array<double>& input, const Array<double>& target, double precision)
: MetaSVM(svm, 1)
{
	setBetaInv(precision);
	train(input, target);
}


bool GaussianProcess::isFeasible()
{
	return (parameter(0) > 0) && (parameter(1) > 0);
}

void GaussianProcess::train(const Array<double>& input, const Array<double>& target)
{
	unsigned int N = input.dim(0);
	this->target = target;

	// store codebook ("support") vectors
	getSVM()->SetTrainingData(input);

	// compute Covariance matrix (6.63) in C. M.  Bishop "Pattern Recognition and Machine Learning", 2007
	unsigned int i, j;
	C.resize(N, N, false);
	CInv.resize(N, N, false);
	for (i = 0; i < N; i++)
	{
		C(i, i) = (*kernel)(input[i], input[i]) + parameter(0);
		for (j = i + 1; j < N; j++) C(i, j) = C(j, i) = (*kernel)(input[i], input[j]);
	}

	// regularize and invert Gram Matrix
	//invertSymm(CInv, C);
	invertSymmPositiveDefinite( CInv, C );

	// compute weights
	for (i = 0; i < N; i++)
	{
		double value = 0.0;
		for (j = 0; j < N; j++) value += CInv(i, j) * target(j, 0);

#ifndef __SOLARIS__
		if (! finite(value))
		{
			throw SHARKEXCEPTION("[GaussianProcess::train] numerical problems");
		}
#endif
		svm->setParameter(i, value);
	}
	svm->setParameter(N, 0.0);   // zero offset
}

double GaussianProcess::operator()(const Array<double>& x)
{
	return getSVM()->model(x);
}

void GaussianProcess::setBetaInv(double precision)
{
	setParameter(0, precision);
}

void GaussianProcess::setSigma(double s)
{
	setParameter(1, s);
}

const Array2D<double>& GaussianProcess::getC()
{
	return C;
}

const Array2D<double>& GaussianProcess::getCInv()
{
	return CInv;
}

const Array<double>& GaussianProcess::getTarget()
{
	return target;
}

double GaussianProcess::Variance(const Array<double>& input, const Array<double>& target)
{
	GaussianProcessVariance variance;
	return variance.error(*this, input, target);
}

double GaussianProcess::Evidence(const Array<double>& input, const Array<double>& target)
{
	GaussianProcessEvidence evidence;
	return evidence.error(*this, input, target);
}


////////////////////////////////////////////////////////////


GaussianProcessVariance::GaussianProcessVariance()
{}

GaussianProcessVariance::~GaussianProcessVariance()
{}


double GaussianProcessVariance::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	GaussianProcess* gp = dynamic_cast<GaussianProcess*>(&model);
	if (gp == NULL) throw SHARKEXCEPTION("[GaussianProcessVariance::error] model must be a GaussianProcess object");

	SVM* svm = gp->getSVM();
	KernelFunction* kernel = svm->getKernel();
	unsigned int i, ic = svm->getExamples();
	Array<double> kv(ic);
	const Array<double>& p = svm->getPoints();
	for (i = 0; i < ic; i++) kv(i) = (*kernel)(p[i], input);
	return (*kernel)(input, input) + gp->getParameter(0) - vecMatVec(kv, gp->getCInv(), kv);
}


////////////////////////////////////////////////////////////


GaussianProcessEvidence::GaussianProcessEvidence()
{}

GaussianProcessEvidence::~GaussianProcessEvidence()
{}


double GaussianProcessEvidence::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	GaussianProcess* gp = dynamic_cast<GaussianProcess*>(&model);
	if (gp == NULL) throw SHARKEXCEPTION("[GaussianProcessEvidence::error] model must be a GaussianProcess object");

	return -evidence(gp);
}

double GaussianProcessEvidence::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	GaussianProcess* gp = dynamic_cast<GaussianProcess*>(&model);
	if (gp == NULL) throw SHARKEXCEPTION("[GaussianProcessEvidence::errorDerivative] model must be a GaussianProcess object");

	double d1;
	Array<double> d2;
	dEvidenceDTheta(gp, d1, d2);
	int d, dim = d2.dim(0);
	derivative.resize(dim + 1, false);
	derivative(0) = -d1;
	for (d=0; d<dim; d++) derivative(d+1) = -d2(d);

	return -evidence(gp);
}

double GaussianProcessEvidence::evidence(GaussianProcess* gp)
{
	SVM* svm = gp->getSVM();
	double e, logdet;
	unsigned int N = svm->getExamples();
	Array2D<double> M;
	Array2D<double> eigVecs;
	Array<double>   eigVals;
	M = gp->getC();
	logdet = logdetsymm(M, eigVecs, eigVals);
	e = -logdet / 2.0 - vecMatVec(gp->getTarget(), 0, gp->getCInv(), gp->getTarget(), 0) / 2.0 - (N / 2.0) * log(2.0 * M_PI);    // (6.69) in C. M.  Bishop "Pattern Recognition and Machine Learning", 2007
	return e;
}

void GaussianProcessEvidence::dEvidenceDTheta(GaussianProcess* gp, double& dEdBI, Array<double>& dEdS)
{
	SVM* svm = gp->getSVM();
	KernelFunction* kernel = svm->getKernel();
	unsigned int N = svm->getExamples();
	Array2D<double> H;
	Array2D<double> D;
	Array<double> dK; // dK/dSigma
	Array<double>   derivative;
	const Array<double>& p = svm->getPoints();

	// 1. compute derivative w.r.t. kernel parameters

	// dC/dSigma = dK/dSigma, where K is the Gram matrix
	unsigned int ell = gp->getC().dim(0);
	unsigned int pp, params = kernel->getParameterDimension();
	dEdS.resize(params, false);
	dK.resize(params, ell, ell, false);
	for (unsigned i = 0; i < N; i++)
	{
		kernel->evalDerivative(p[i], p[i], derivative);
		for (pp=0; pp<params; pp++) dK(pp, i, i) = derivative(pp);
		for (unsigned j = i + 1; j < N; j++)
		{
			kernel->evalDerivative(p[i], p[j], derivative);
			for (pp=0; pp<params; pp++) dK(pp, i, j) = dK(pp, j, i) = derivative(pp);
		}
	}
	const Array2D<double>& CInv = gp->getCInv();
	for (pp=0; pp<params; pp++)
	{
		matMat(H, CInv, dK[pp]);
		dEdS(pp) = -trace(H) / 2.0;
		matMat(D, H, CInv);
		dEdS(pp) += vecMatVec(gp->getTarget(), 0, D, gp->getTarget(), 0) / 2.0;
	}

	// 2. compute derivative w.r.t. betaInv

	// dC/dBetaInv = I
	dEdBI = -trace(gp->getCInv()) / 2.0;

	matMat(D, gp->getCInv(), gp->getCInv());
	dEdBI += vecMatVec(gp->getTarget(), 0, D, gp->getTarget(), 0) / 2.0;
}

