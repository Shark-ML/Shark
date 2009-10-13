//===========================================================================
/*!
*  \file ErrorFunction.h
*
*  \brief Base class of all error measures.
*
*  \par Copyright (c) 1999-2005:
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
*  \par Project:
*      Shark
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


#ifndef _ErrorFunction_H_
#define _ErrorFunction_H_


#include <SharkDefs.h>
#include <ReClaM/Model.h>


/*!
*
* \brief Base class of all error measures.
*
* ReClaM provides the three base classes Model, ErrorFunction and
* Optimizer which make up the ReClaM framework for solving regression
* and classification tasks. This design overrides the ModelInterface
* design which is kept for downward compatibility.<BR>
* The ErrorFunction class encapsulates an error function operating on
* any model defined by the Model class.
*
* An ErrorFunction computes a risk, that is, a loss function over a
* set of training patterns. It takes a model, a set of input patterns
* and a set of corresponding targets as its input, and outputs a
* single double value (the error).
*
* The #input array can be expected to be two dimensional, indexed by
* the patterns and the input space dimension. In the general case,
* the target will be two dimensional, too, indexed by the patterns
* and the model output dimension. However, ErrorFunctions should be
* aware of the case that the underlying Model is a function. In this
* case model #output and #target array may lack the output dimension,
* that is, they may be one dimensional, indexed by the patterns.
* It possible (and meaningful), ErrorFunctions should handle both
* cases. If not, they should throw an exception in the unhandled case.
*
*
*  \author  T. Glasmachers, C. Igel
*  \date    2005
*/
class ErrorFunction
{
public:
	//! Constructor
	ErrorFunction();

	//! Destructor
	virtual ~ErrorFunction();


	//===========================================================================
	/*!
	*  \brief Error of the model. 
	*
	*  This method calculates the error of the model.
	*  #target is compared with the model's output for stimulus #input.
	*
	*      \param  model Model to use for the computation.
	*      \param  input Vector of input values.
	*      \param  target Vector of output values.
	*      \return The error of the model.
	*
	*  \author  C. Igel
	*  \date    1999
	*
	*  \par Changes
	*      none
	*
	*  \par Status
	*      stable
	*
	*/
	virtual double error(Model& model, const Array<double>& input, const Array<double>& target) = 0;

	//===========================================================================
	/*!
	*  \brief Calculates the field derivative, the derivative of the error
	*  \f$E\f$ with respect to the model parameters.
	*
	*  Calculates the derivates of the error \f$E\f$ with respect to the
	*  model parameters and stores the result in the elements of the array
	*  #derivative. As a default implementation, the derivatives are estimated
	*  numerically by using small variations #epsilon:
	*
	*  \f[\frac{\partial E}{\partial \omega_i} = 
	*      \frac{E(\omega_i + \epsilon) - E(\omega_i)}{\epsilon} 
	*      + O(\epsilon) \f]
	*
	*  However, to improve the speed, it is adviserable to overload this
	*  method by a more sophisticated calculation, depending on the
	*  method #error. Nevertheless, this
	*  default implementation can be used to check new implementations of
	*  the derivative of the error with respect to the model parameters.
	*
	*  Usually, as a byproduct of the calculation of the derivative one gets the
	*  the error \f$E\f$ itself very efficiently. Therefore, the method #errorDerivative
	*  gives back this value.
	*
	*
	*      \param  model Model to use for the computation.
	*      \param  input Vector of input values.
	*      \param  target Vector of output values.
	*      \param  derivative Vector of partial derivatives.
	*      \return The error \em E
	*
	*
	*  \author  C. Igel
	*  \date    1999
	*
	*  \par Changes
	*      3/2006 Tobias Glasmachers: step size epsilon has bacome a fraction
	*             of the current parameter value
	*
	*  \par Status
	*      stable
	*
	*/
	virtual double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//===========================================================================
	/*!
	*  \brief Sets the value of #epsilon.
	*
	*  The value of #epsilon is utilized for a numeric estimation of the
	*  derivative of the error, with respect to the model parameters.
	*  #epsilon should be choosen as a small value. The default value
	*  is #epsilon = 1e-8.
	*
	*      \param  eps New value for epsilon.
	*      \return None.
	*
	*  \author  C. Igel
	*  \date    1999
	*
	*  \par Changes
	*      none
	*
	*  \par Status
	*      stable
	*
	*/
	inline void setEpsilon(double eps)
	{
		epsilon = eps;
	};

protected:
	//! Precision parameter for the numerical error gradient approximation.
	double epsilon;
};


#endif

