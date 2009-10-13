//===========================================================================
/*!
 *  \file RadiusMargin.h
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


#ifndef _RadiusMargin_H_
#define _RadiusMargin_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/Svm.h>
#include <ReClaM/QuadraticProgram.h>


//!
//! \brief Squared Radius-Margin-Quotient
//!
//! \author T. Glasmachers
//! \date 2006
//!
//! \par
//! The Radius-Margin-Quotient \f$ R^/\gamma^2 \f$ is a well
//! known quantity in statistical learning theory which can be
//! turned into a generalization bound after normalization.
//! Its minimization has been proposed for SVM model selection.
//!
//! \par
//! The Radius-Margin-Quotient depends on the kernel parameters
//! as well as the SVM regularization parameter C. However, as
//! the derivative of the \f$ \alpha \f$ vector w.r.t. C is hard
//! to compute (in fact, it involves the inverse of the matrix
//! \f$ K_{ij} = k(x_i, x_j) \f$ which is considered
//! to be too large to fit into memory), the #errorDerivative
//! member only computes the derivative w.r.t the kernel
//! parameters. However, if the 2-norm slack penalty formuation
//! is used, C can be interpreted as a kernel matrix parameter
//! and the derivative can be computed easily.
//!
//! \par
//! The underlying quadratic program solver may be configured
//! to use a maximal number of iterations. Thus it is not
//! guaranteed to find the global optimum with prespecified
//! accuracy. In this case a radius margin quotient of
//! 1e100 is returned and the derivative is set to zero.
//!
class RadiusMargin : public ErrorFunction
{
public:
	//! Constructor
	RadiusMargin();

	//! Destructor
	~RadiusMargin();


	//! Computes the radius margin quotient.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computes the radius margin quotient and its derivatives
	//! as proposed by Chapelle, Vapnik, Bousquet and Mukherjee in
	//! "Choosing Multiple Parameters for Support Vector Machines",
	//! sections 6.1 and 6.2. Please note that in the 1-norm slack
	//! penalty case the derivative w.r.t. C is returned as zero,
	//! which is actually not the case. The computation of this
	//! derivative would require the inversion of the kernel gram
	//! matrix restricted to the support vectors, which is
	//! considered infeasible. It would be much cheapter to compute
	//! a numerical estimate.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//! set the maximum number of iterations
	//! for the quadratic program solver
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	//! return the number of iterations last
	//! used by the quadratic program solver
	inline SharkInt64 iterations()
	{
		return iter;
	}

protected:
	//! Helper function calling the quadratic problem solver
	//! to obtain the coefficients \f$\alpha\f$ and \f$\beta\f$.
	//! The function returns the squared radius or -1.0 to
	//! indicate that the solver did not find the optimum.
	double solveProblems(SVM* pSVM, double Cplus, double Cminus, const Array<double>& input, const Array<double>& target, Array<double>& alpha, Array<double>& beta, bool norm2);

	//! maximum number of #C_Solver iterations
	SharkInt64 maxIter;

	//! last number if #C_Solver iterations
	SharkInt64 iter;
};


#endif

