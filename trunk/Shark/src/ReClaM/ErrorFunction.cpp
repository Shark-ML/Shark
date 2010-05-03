//===========================================================================
/*!
*  \file ErrorFunction.cpp
*
*  \brief Base class of all error measures.
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
*  \author  T. Glasmachers
*  \date    2005
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

#include <ReClaM/ErrorFunction.h>
#include <math.h>


ErrorFunction::ErrorFunction()
{
	epsilon = 1e-2;
}

ErrorFunction::~ErrorFunction()
{
}


double ErrorFunction::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	int p, pc = model.getParameterDimension();
	derivative.resize(pc, false);

	double ret = error(model, input, target);
	double temp, eps, e;
	for (p = 0; p < pc; p++)
	{
		temp = model.getParameter(p);
		eps = epsilon * temp;
		if (fabs(eps) < fabs(0.1 * epsilon)) eps = epsilon;
		model.setParameter(p, temp + eps);
		e = error(model, input, target);
		derivative(p) = (e - ret) / eps;
		model.setParameter(p, temp);
	}

	return ret;
}

