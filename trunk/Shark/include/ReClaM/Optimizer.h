//===========================================================================
/*!
*  \file Optimizer.h
*
*  \brief Base class of all optimizers.
*
*  \author  T. Glasmachers, C. Igel
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


#ifndef _Optimizer_H_
#define _Optimizer_H_


#include <ReClaM/ModelWithErrorFunction.h>


/*!
 *  \brief Base class of all optimizers.
 *
 *  \par
 *  ReClaM provides the three base classes Model, ErrorFunction and
 *  Optimizer which make up the ReClaM framework for solving regression
 *  and classification tasks. This design overrides the ModelInterface
 *  design which is kept for downward compatibility.<BR>
 *  The Optimizer class encapsulates data driven a stepwise optimization
 *  technique with the goal to minimize an ErrorFunction computed on
 *  a Model.
 */
class Optimizer
{
public:
	//! Constructor
	Optimizer();

	//! Destructor
	virtual ~Optimizer();


	//! basic initialization with default parameters
	virtual void init(Model& model) = 0;

	//===========================================================================
	/*!
	*  \brief Performes one optimization step, for example a gradient
	*  descent step or an evolution cycle.
	*
	*
	*      \param  model          Model to use for the computation.
	*      \param  errorfunction  ErrorFunction on which decisions are based.
	*      \param  input          Vector of input values.
	*      \param  target         Vector of output values.
	*      \return The error \em E returned by the ErrorFunction.
	*
	*/
	virtual double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target) = 0;


	//===========================================================================
	/*!
	*  \brief Performes one optimization step, for example a gradient
	*  descent step or an evolution cycle.
	*
	*
	*      \param  me             Model and ErrorFunction to use for the computation.
	*      \param  errorfunction  ErrorFunction on which decisions are based.
	*      \param  input          Vector of input values.
	*      \param  target         Vector of output values.
	*      \return The error \em E returned by the ErrorFunction.
	*
	*/
	virtual double optimize(ModelWithErrorFunction& me, const Array<double>& input, const Array<double>& target) {
		return optimize(me, me, input, target);
	}

	//===========================================================================
	/*!
	*  \brief Performes one optimization step, for example a gradient
	*  descent step or an evolution cycle.
	*
	*
	*      \param  me             Model and ErrorFunction to use for the computation.
	*      \param  errorfunction  ErrorFunction on which decisions are based.
	*      \param  input          Vector of input values.
	*      \return The error \em E returned by the ErrorFunction.
	*
	*/
	virtual double optimize(ModelWithErrorFunction& me, const Array<double>& input) {
		return optimize(me, me, input, input);
	}


};


#endif

