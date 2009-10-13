//===========================================================================
/*!
*  \file ValidationError.h
*
*  \brief Compute the error on a hold out set
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


#include <vector>

#include <Rng/GlobalRng.h>
#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/Optimizer.h>


//!
//! \brief Error on a hold out set
//!
//! \par
//! First, the data are split into training and validation datasets.
//! Then a model is trained on the training subset with a given
//! base error function and a given optimizer for a predefined
//! number of iterations. Finally the error on the validation set is
//! reported.
//!
class ValidationError : public ErrorFunction
{
public:
	//! Constructor
	//!
	//! \param  base             error function for machine training and evaluation of the validation set
	//! \param  opt              optimizer for machine training
	//! \param  iter             number of optimization iterations
	//! \param  holdOutFraction  fraction of the data in the hold out set
	ValidationError(ErrorFunction* base, Optimizer* opt, int iter, double holdOutFraction = 0.2);

	//! Descructor
	~ValidationError();


	//! Hold-out error computation:
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Hold-out error computation:
	//! The data are split into training and validation datasets.
	//! The model is trained on the training subset with the given
	//! base error function and the given optimizer for the
	//! predefined number of iterations.
	//! The method reports the error on the validation set, including
	//! its derivative. Make sure that the derivative computation for
	//! the base error is implemented before using this function.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

protected:
	//! error function for training and evaluation
	ErrorFunction* baseError;
	
	//! optimizer for machine training
	Optimizer* optimizer;

	//! number of training iterations
	int iterations;

	//! fraction of the data used for evaluation
	double holdOut;
};
