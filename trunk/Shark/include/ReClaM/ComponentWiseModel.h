//===========================================================================
/*!
*  \file ComponentWiseModel.h
*
*  \brief The ComponentWiseModel encapsulates the component wise application of a base model.
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


#ifndef _ComponentWiseModel_H_
#define _ComponentWiseModel_H_


#include <ReClaM/Model.h>
#include <vector>


//!
//! \brief The ComponentWiseModel encapsulates the component wise application of a base model.
//!
//! \par
//! The component wise application of a model to data is a mapping of the form
//! \f$ f(x_1, \dots, x_n) = (g(x_1), \dots, g(x_n)) \f$, that is, the model f
//! makes its predictions by applying the model g to each component of the input
//! data. Of course, the parameters of f and g coincide.
//!
class ComponentWiseModel : public Model
{
public:
	//! Constructor
	ComponentWiseModel(Model* pBase, int numberOfCopies);

	//! Destructor
	~ComponentWiseModel();


	//! Evaluate the ComponentWise model
	void model(const Array<double>& input, Array<double> &output);

	//! Evaluate the ComponentWise model and compute its derivative
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! Evaluate the ComponentWise model and compute its derivative
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);

	//! Check whether the parameters define a valid model
	bool isFeasible();

protected:
	//! number of base model copies
	int copies;

	//! single base model
	Model* base;
};


#endif

