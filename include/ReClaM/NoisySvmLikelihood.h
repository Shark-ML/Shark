//===========================================================================
/*!
 *  \file NoisySvmLikelihood.h
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


#ifndef _NoisySvmLikelihood_H_
#define _NoisySvmLikelihood_H_


#include <ReClaM/ErrorFunction.h>


//!
//! \brief model selection objective for SVMs
//!
class NoisySvmLikelihood : public ErrorFunction
{
public:
	//! Constructor
	//! \param  trainFraction  fraction of the data used for SVM training
	NoisySvmLikelihood(double trainFraction = 0.8);

	//! Destructor
	~NoisySvmLikelihood();


	//! computation of the negative log likelihood
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! computation of the negative log likelihood
	//! and its derivatives w.r.t. the model parameters
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

protected:
	//! fraction of the data used for training
	double trainFraction;
};


#endif
