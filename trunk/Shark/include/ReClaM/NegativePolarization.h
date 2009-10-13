/*!
 *  \file NegativePolarization.h
 *
 *  \brief Implementation of the negative Kernel Polarization
 *  	   Measure, that is, Kernel Target Alignment without normalization.
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


#ifndef _Polarization_H_
#define _Polarization_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/KernelFunction.h>
#include <ReClaM/Svm.h>


/*!
 *  \brief Implementation of the negative Kernel Polarization
 *  	   Measure, that is, Kernel Target Alignment without normalization.
 *
 *  \par Kernel polarization measures how good a kernel fits
 *  a binary classification training set. It is NOT invariant
 *  under kernel rescaling. To turn it into an error function
 *  (i.e., to make minimization meaningful) the negative
 *  polarization is implemented.
 *
 *  \par The polarization has two important properties: It is
 *  differentiable and independent of the actual classifier.
 *
 *  \par The class NegativePolarization accepts two
 *  kinds of models: KernelFunction derived classes and C_SVM
 *  with 2-norm slack penalty.
 *  In the first case, only the kernel parameters are considered
 *  in the derivative computation. In the second case,
 *  in addition the derivatives w.r.t. the parameters \f$ C_+ \f$
 *  and \f$ C_- \f$ are computed.
 *
 *  \par Please note that is the vast majority of cases it is a
 *  good idea to consider the kernel target alignment class
 *  #NegativeKTA instead.
 */
class NegativePolarization : public ErrorFunction
{
public:
	//! Constructor
	NegativePolarization();

	//! Destructor
	~NegativePolarization();


	//! Computes the negative kernel polarization between
	//! the target and the kernel function output on the input.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computes the negative kernel polarization between
	//! the target and the kernel function output on the input.
	//! The partial derivatives of the negative polarization w.r.t.
	//! the model parameters are returned in the derivative parameter.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);
};


#endif

