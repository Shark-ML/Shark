//===========================================================================
/*!
 *  \file GaussianProcess.h
 *
 *  \brief Gaussian Process header
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


#ifndef _GaussianProcess_H_
#define _GaussianProcess_H_

#include <Array/Array.h>
#include <Array/Array2D.h>
#include <ReClaM/Svm.h>


//!
//! \brief Gaussian Process
//!
//! \par
//! Encapsulation of a Gaussian process, viewed as an
//! #SVM regression training scheme.
//!
//! \par
//! The #GaussianProcess model holds a regularization parameter
//! and the kernel parameters, making up the model hyper parameters.
//! The classes #GaussianProcessVariance and #GaussianProcessEvidence
//! can be used for the adaptation of these hyper parameters.
//!
//! The #GaussianProcess model can either be trained directly or by a
//! #SVM_Optimizer object.
//!
class GaussianProcess : public MetaSVM
{
public:
	//! Constructor
	//! @param precision the regularization parameter \f$ \beta^{-1} \f$
	GaussianProcess(SVM* svm, double precision = 0.0);

	//! Constructor
	//! @param precision the regularization parameter \f$ \beta^{-1} \f$
	GaussianProcess(SVM* svm, const Array<double>& input, const Array<double>& target, double precision);


	//! Check the feasibility of all hyper parameters.
	bool isFeasible();

	//! Train the underlying #SVM on the given training data
	//! using the current hyper parameter configuration.
	//! This function will also be called when #GaussianProcess model is
	//! optimized by an #SVM_Optimizer.
	void train(const Array<double>& input, const Array<double>& target);

	//! Do a model prediction, that is, use the underlying
	//! #SVM to process the given sample. This method is
	//! equivalent to calling the underlying #SVM object.
	double operator()(const Array<double>& x);

	//! Set the regularization parameter \f$ \beta^{-1} \f$
	void setBetaInv(double precision);

	//! Set the kernel parameter \f$ \sigma \f$, assuming
	//! an underlying #NormalizedRBFKernel kernel object.
	//! In general, this method sets the first kernel
	//! parameter to the specified value.
	void setSigma(double s);

	//! Return the regularized kernel matrix
	const Array2D<double>& getC();

	//! Return the inverse of the regularized kernel matrix
	const Array2D<double>& getCInv();

	//! Return the regression target values
	const Array<double>& getTarget();

	//! Return the variance of the Gaussian
	//! process given training data
	double Variance(const Array<double>& input, const Array<double>& target);

	//! Return the evidence of the Gaussian
	//! process given training data
	double Evidence(const Array<double>& input, const Array<double>& target);

protected:
	//! regularized kernel matrix,
	//! valid after a call to #train()
	Array2D<double> C;

	//! inverse of the regularized kernel matrix,
	//! valid after a call to #train()
	Array2D<double> CInv;

	//! regression targets,
	//! valid after a call to #train()
	Array<double> target;
};


//!
//! \brief variance of a Gaussian process
//!
//! \par
//! The #GaussianProcessVariance class is an
//! ErrorFunction and can thus be used as an
//! optimization target. Usually it will only
//! be used for monitoring.
//!
class GaussianProcessVariance : public ErrorFunction
{
public:
	//! Constructor
	GaussianProcessVariance();

	//! Destructor
	~GaussianProcessVariance();

	//! Computation of the Gaussian process variance.
	//! \param  model  A #GaussianProcess model must be provided, otherwise an exception will be thrown
	double error(Model& model, const Array<double>& input, const Array<double>& target);
};


//!
//! \brief negative evidence of a Gaussian process
//!
//! \par
//! The #GaussianProcessEvidence class serves as an
//! #ErrorFunction, that is, any #Optimizer can be
//! used for its minimization (aiming at evidence
//! maximization).
//!
class GaussianProcessEvidence : public ErrorFunction
{
public:
	//! Constructor
	GaussianProcessEvidence();

	//! Destructor
	~GaussianProcessEvidence();

	//! Computation of the negative Gaussian process evidence.
	//! \param  model  A #GaussianProcess model must be provided, otherwise an exception will be thrown
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computation of the derivative of the negative Gaussian process evidence.
	//! \param  model  A #GaussianProcess model must be provided, otherwise an exception will be thrown
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

protected:
	//! Computation of the evidence
	double evidence(GaussianProcess* gp);

	//! Computation of the derivative of the evidence
	void dEvidenceDTheta(GaussianProcess* gp, double &dEdBI, Array<double>& dEdS);
};


#endif

