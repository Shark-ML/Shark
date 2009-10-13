//===========================================================================
/*!
*  \file SigmoidModel.h
*
*  \brief sigmoidal functions
*
*  \par sigmoid functions \f$ R \rightarrow [0, 1] \f$
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \par Copyright (c) 1999-2006:
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

#ifndef _SigmoidModel_H_
#define _SigmoidModel_H_


#include <ReClaM/Model.h>


//! \brief Standard sigmoid function with two parameters
//!
//! \par
//! This model maps the reals to the unit interval by the sigmoid function
//! \f$ f_{(A, B)}(x) = \frac{1}{1 + \exp(Ax+B)} \f$.
class SigmoidModel : public Model
{
public:
	//! Constructor
	SigmoidModel(double A = -1.0, double B = 0.0);

	//! Destructor
	~SigmoidModel();


	//! get the maximal steepness
	inline double get_A()
	{
		return parameter(0);
	}

	//! get the shift
	inline double get_B()
	{
		return parameter(1);
	}

	//! apply the model
	void model(const Array<double>& input, Array<double>& output);

	//! compute the derivative of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! compute the model output and its derivative
	//! of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);
};


//! \brief Simple sigmoid function with one parameter
//!
//! \par
//! This model maps the reals to the unit interval by the sigmoid function
//! \f$ f_s(x) =  \frac{1}{2} \frac{st}{1+s|t|} + \frac{1}{2} \f$.
class SimpleSigmoidModel : public Model
{
public:
	//! Constructor
	SimpleSigmoidModel(double s = 1.0);

	//! Destructor
	~SimpleSigmoidModel();


	//! get the parameter
	inline double get_s()
	{
		return parameter(0);
	}

	//! apply the model
	void model(const Array<double>& input, Array<double> &output);

	//! compute the derivative of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! compute the model output and its derivative
	//! of the model output w.r.t. the parameters
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);

	//! check if the parameter s is positive
	bool isFeasible();
};


#endif

