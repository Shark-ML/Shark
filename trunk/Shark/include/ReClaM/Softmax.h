//===========================================================================
/*!
*  \file Softmax.h
*
*  \brief soft-max function mapping \f$ R^n \f$ to a probability
*
*  \author  T. Glasmachers
*  \date    2010
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

#ifndef _Softmax_H_
#define _Softmax_H_


#include <ReClaM/Model.h>


//!
//! \brief Softmax function
//!
//! \par
//! Squash an n-dimensional real vector space
//! to the (n-1)-dimensional probability simplex.
//!
class Softmax : public Model
{
public:
	//! Constructor
	Softmax(unsigned int dim);

	//! Destructor
	~Softmax();


	//! apply the model
	void model(const Array<double>& input, Array<double>& output);

	//! compute the derivative of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! compute the model output and its derivative
	//! of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);
};


#endif
