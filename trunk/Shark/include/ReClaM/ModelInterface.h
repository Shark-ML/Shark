//===========================================================================
/*!
*  \file ModelInterface.h
*
*  \brief [DEPRECATED] Realizes the communication between the different modules.
*
*  <b>ATTENTION: THIS FILE IS DEPRECATED!</b><br>
*  It is provided for downward compatibility with Shark versions
*  up to 1.4.x only. Programmers are discouraged from using it.
*  A new design is defined through the classes #Model,
*  #ErrorFunction, and #Optimizer.
*  This class is a workaround which defines some dummy functions
*  and data members that mimic the behaviour of the original
*  ModelInterface class. It's description follows:<br>
*
*  To provide flexibility ReClaM offers different modules that can
*  be put together to form an environment for solving regression
*  and classification tasks.<BR>
*  The choice of three module types is necessary: A parametric model,
*  an error function module and an optimization algorithm model.
*  All modules exchange parameters and other information, so the task
*  of this class is to offer this communication.
*
*  \author  C. Igel
*  \date    1999
*
*  \par Copyright (c) 1999-2001:
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

#ifndef _ModelInterface_H_
#define _ModelInterface_H_


#include "Model.h"
#include "ErrorFunction.h"


/*!
*
*  \brief [DEPRECATED] Realizes the communication between the different modules.
*
*  <b>ATTENTION: THIS CLASS IS DEPRECATED!</b><br>
*  It is provided for downward compatibility with Shark versions
*  up to 1.4.x only. Programmers are discouraged from using it.
*  A new design is defined through the classes #Model,
*  #ErrorFunction and #Optimizer.
*  This class is a workaround which defines some dummy functions
*  and data members that mimic the behaviour of the original
*  ModelInterface class.<br>
*/
class ModelInterface : public ErrorFunction, public Model
{
public:
	//! The constructor defines references to those variables
	//! whose names have changed in the context of the ReClaM
	//! re-design replacing #ModelInterface by #Model.
	ModelInterface()
			: w(Model::parameter)
	{}

	//! Descructor
	virtual ~ModelInterface()
	{}


	//! Model output - this call is passed through to #Model.
	virtual void dmodel(const Array<double>& input)
	{
		((Model*)this)->modelDerivative(input, dmdw);
	}

	//! Derivative of the model output - this call is passed through to #Model.
	virtual void dmodel(const Array<double>& input, Array<double>& output)
	{
		((Model*)this)->modelDerivative(input, output, dmdw);
	}

	//! Error evaluation - this call is passed through to #ErrorFunction.
	virtual double error(const Array<double>& input, const Array<double>& target)
	{
		return ((ErrorFunction*)this)->error(*(Model*)this, input, target);
	}

	//! Derivative of the error - this call is passed through to #ErrorFunction.
	virtual double derror(const Array<double>& input, const Array<double>& target, bool bReturnError = true)
	{
		return ((ErrorFunction*)this)->errorDerivative(*(Model*)this, input, target, dedw);
	}

protected:
	//! Reference to #parameter
	Array<double>& w;

	//! Derivative of the model output w.r.t #w, computed by #dmodel
	Array<double> dmdw;

	//! Derivative of the error w.r.t. #w, computed by #derror
	Array<double> dedw;
};


#endif

