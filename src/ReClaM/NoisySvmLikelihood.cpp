//===========================================================================
/*!
 *  \file NoisySvmLikelihood.cpp
 *
 *  \brief model selection objective function for SVMs
 *
 *  \author  T. Glasmachers
 *  \date    2008
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


#include <Rng/GlobalRng.h>
#include <ReClaM/NoisySvmLikelihood.h>
#include <ReClaM/Svm.h>
#include <ReClaM/SigmoidModel.h>
#include <ReClaM/NegativeLogLikelihood.h>
#include <ReClaM/Rprop.h>

#include <vector>
#include <algorithm>


// choose exactly one of these:

// exponential decaying sigmoid
#define SIG_E

// polynomial decaying sigmoid
// #define SIG_P



////////////////////////////////////////////////////////////


NoisySvmLikelihood::NoisySvmLikelihood(double trainFraction)
{
	RANGE_CHECK(trainFraction > 0.0 && trainFraction < 1.0);

	this->trainFraction = trainFraction;
}

NoisySvmLikelihood::~NoisySvmLikelihood()
{
}


double NoisySvmLikelihood::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);

	// check the model type
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL || csvm->is2norm()) throw SHARKEXCEPTION("[NoisySvmLikelihood::error] model must be a 1-norm C-SVM.");

	// randomly split the data
	int i, ic = input.dim(0);
	int dim = input.dim(1);
	int train_c = (int)(trainFraction * ic);
	int test_c = ic - train_c;
	SIZE_CHECK(train_c > 0 && test_c > 0);
	Array<double> train_d(train_c, dim);
	Array<double> train_l(train_c, 1);
	Array<double> test_d(test_c, dim);
	Array<double> test_l(test_c, 1);
	std::vector<int> permutation(ic);
	for (i=0; i<ic; i++) permutation[i] = i;
	for (i=0; i<ic; i++)
	{
		int j = Rng::discrete(0, ic-1);
		int tmp = permutation[j];
		permutation[j] = permutation[i];
		permutation[i] = tmp;
	}
	for (i=0; i<train_c; i++)
	{
		train_d[i] = input[permutation[i]];
		train_l[i] = target[permutation[i]];
	}
	for (i=0; i<test_c; i++)
	{
		test_d[i] = input[permutation[train_c + i]];
		test_l[i] = target[permutation[train_c + i]];
	}

	// train the SVM
	SVM* svm = csvm->getSVM();
	SVM_Optimizer opt;
	opt.init(*csvm);
	opt.optimize(*svm, train_d, train_l);

	// predict the validation data
	Array<double> z(test_c, 1);
	svm->model(test_d, z);

	// train a sigmoid on the validation data
#ifdef SIG_E
	SigmoidModel sigmoid;
#endif
#ifdef SIG_P
	SimpleSigmoidModel sigmoid;
#endif
	NegativeLogLikelihood nll;
	IRpropPlus rprop;
	rprop.init(sigmoid);
	for (i=0; i<100; i++)
	{
		rprop.optimize(sigmoid, nll, z, test_l);
#ifdef SIG_E
		sigmoid.setParameter(1, 0.0);
		if (sigmoid.getParameter(0) > 0.0) sigmoid.setParameter(0, 0.0);
#endif
#ifdef SIG_P
		if (sigmoid.getParameter(0) < 0.0) sigmoid.setParameter(0, 0.0);
#endif
	}

	// return the best negative log likelihood
	return nll.error(sigmoid, z, test_l);
}

double NoisySvmLikelihood::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	SIZE_CHECK(input.ndim() == 2);
	SIZE_CHECK(target.ndim() == 2);

	// check the model type
	C_SVM* csvm = dynamic_cast<C_SVM*>(&model);
	if (csvm == NULL || csvm->is2norm()) throw SHARKEXCEPTION("[NoisySvmLikelihood::errorDerivative] model must be a 1-norm C-SVM.");

	// randomly split the data
	int i, ic = input.dim(0);
	int dim = input.dim(1);
	int train_c = (int)(trainFraction * ic);
	int test_c = ic - train_c;
	SIZE_CHECK(train_c > 0 && test_c > 0);
	Array<double> train_d(train_c, dim);
	Array<double> train_l(train_c, 1);
	Array<double> test_d(test_c, dim);
	Array<double> test_l(test_c, 1);
	std::vector<int> permutation(ic);
	for (i=0; i<ic; i++) permutation[i] = i;
	for (i=0; i<ic; i++)
	{
		int j = Rng::discrete(0, ic-1);
		int tmp = permutation[j];
		permutation[j] = permutation[i];
		permutation[i] = tmp;
	}
	for (i=0; i<train_c; i++)
	{
		train_d[i] = input[permutation[i]];
		train_l[i] = target[permutation[i]];
	}
	for (i=0; i<test_c; i++)
	{
		test_d[i] = input[permutation[train_c + i]];
		test_l[i] = target[permutation[train_c + i]];
	}

	// train the SVM
	SVM* svm = csvm->getSVM();
	SVM_Optimizer opt;
	opt.init(*csvm);
	opt.optimize(*svm, train_d, train_l);

	// predict the validation data
	Array<double> z(test_c, 1);
	svm->model(test_d, z);

	// train a sigmoid on the validation data
#ifdef SIG_E
	SigmoidModel sigmoid;
#endif
#ifdef SIG_P
	SimpleSigmoidModel sigmoid;
#endif
	NegativeLogLikelihood nll;
	IRpropPlus rprop;
	rprop.init(sigmoid);
	for (i=0; i<100; i++)
	{
		rprop.optimize(sigmoid, nll, z, test_l);
#ifdef SIG_E
		sigmoid.setParameter(1, 0.0);
		if (sigmoid.getParameter(0) > 0.0) sigmoid.setParameter(0, 0.0);
#endif
#ifdef SIG_P
		if (sigmoid.getParameter(0) < 0.0) sigmoid.setParameter(0, 0.0);
#endif
	}

	// compute the derivative
	Array<double> p(test_c, 1);
	sigmoid.model(z, p);

	int b, bc = csvm->getParameterDimension();
	derivative.resize(bc, false);
	derivative = 0.0;
	Array<double> dz_dtheta;
	csvm->PrepareDerivative();
	for (i=0; i<test_c; i++)
	{
		// compute the derivative of the negative log likelihood
		double dL_dp;
		if (test_l(i, 0) > 0.0) dL_dp = -1.0 / p(i, 0);
		else dL_dp = -1.0 / (p(i, 0) - 1.0);

		// compute the derivative of the sigmoid
#ifdef SIG_E
		double dp_dz = - sigmoid.getParameter(0) * p(i, 0) * (1.0 - p(i, 0));
#endif
#ifdef SIG_P
		double x = sigmoid.getParameter(0) * p(i, 0);
		double N = 1.0 + fabs(x);
		double dp_dz = sigmoid.getParameter(0) / (N * N);
#endif

		// compute the derivative of the SVM
		csvm->modelDerivative(test_d[i], dz_dtheta);

		// total derivative = partial derivative
		for (b=0; b<bc; b++) derivative(b) += dL_dp * dp_dz * dz_dtheta(0, b);
	}

	// return the best negative log likelihood
	return nll.error(sigmoid, z, test_l);
}
