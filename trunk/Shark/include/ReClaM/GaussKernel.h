//===========================================================================
/*!
 *  \file GaussKernel.h
 *
 *  \brief Gauss kernels with adaptive covariance matrices
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

#ifndef _GaussKernel_H_
#define _GaussKernel_H_


#include <SharkDefs.h>
#include <ReClaM/KernelFunction.h>
#include <LinAlg/LinAlg.h>


//! \brief Guassian Kernel with independent scaling of every axis
//!
//! \f$ k(x, z) = exp(- (x-z)^T D^T D (x-z) ) \f$ with diagonal matrix D
//!
//! In case computation speed matters, it is equivalent
//! to linearly transform the input data according to
//! the diagonal matrix D before applying the standard
//! #RBFKenerl with \f$ \gamma = 1 \f$.
class DiagGaussKernel : public KernelFunction
{
public:
	//! Constructor
	DiagGaussKernel(int dim, double gamma = 1.0);

	//! Destructor
	~DiagGaussKernel();


	//! Evaluates the kernel function.
	double eval(const Array<double>& x1, const Array<double>& x2) const;

	//! Evaluates the kernel function and computes
	//! its derivatives w.r.t. the kernel parameters.
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	//! compute the single RBF coefficient gamma, which is
	//! in this context best defined as the determinant of
	//! the matrix M to 2/d, where d is the input space
	//! dimension.
	double computeGamma();
};


//! \brief General Guassian Kernel
//!
//! \f$ k(x, z) = exp(- (x-z)^T M^T M (x-z) ) \f$ with symmetric matrix M
//!
//! In case computation speed matters, it is equivalent
//! to linearly transform the input data according to
//! the matrix M before applying the standard
//! #RBFKenerl with \f$ \gamma = 1 \f$.
//! This matrix can be determined calling #getTransformation.
class GeneralGaussKernel : public KernelFunction
{
public:
	//! Constructor
	GeneralGaussKernel(int dim, double gamma = 1.0);

	//! Constructor
	//! \param  symmetricTransformation  symmetric positive definite inverse square root of the covariance of the Gaussian
	GeneralGaussKernel(const Array2D<double>& symmetricTransformation);

	//! Destructor
	~GeneralGaussKernel();


	//! Evaluates the kernel function.
	double eval(const Array<double>& x1, const Array<double>& x2) const;

	//! Evaluates the kernel function and computes
	//! its derivatives w.r.t. the kernel parameters.
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	//! This method fills in the quadratic matrix trans
	//! with the linear transformation the kernel applies
	//! to the input data before applying the standard
	//! Gaussian kernel \f$ k(x, z) = exp(-(x-z)^2) \f$.
	void getTransformation(Array<double>& trans);

	//! compute the single RBF coefficient gamma, which is
	//! in this context best defined as the determinant of
	//! the matrix M to 2/d, where d is the input space
	//! dimension.
	double computeGamma();

protected:
	//! compute the index into the parameter vector
	inline static int index(int i, int j)
	{
		return (i < j) ? j*(j + 1) / 2 + i : i*(i + 1) / 2 + j;
	}
};


#endif
