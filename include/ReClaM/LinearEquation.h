//===========================================================================
/*!
*  \file LinearEquation.h
*
*  \brief Model and Error Function for the iterative approximate
*         solution of a linear system
*
*  \author  T. Glasmachers
*  \date    2007
*
*  \par Copyright (c) 1999-2007:
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


#ifndef _LinearEquationSolver_H_
#define _LinearEquationSolver_H_


#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>


//! 
//! \brief Model and Error Function for the iterative approximate
//!        solution of a linear system
//!
//! \par
//! The LinearEquation class takes a matrix M and a vector v
//! as arguments to its constructor, representing the equation
//! \f$ M x = v \f$.
//! The model represents the vector x, while the error function
//! encapsulates the quadratic term
//! \f$ \|M x - v\|^2 \f$.
//! With this error function any ReClaM optimizer can be used
//! to iteratively solve the linear system.
//!
//! \par
//! This class is useful in cases where the inverse of the matrix
//! M is hard to compute, either because the conditioning number
//! or the dimension of M is large, leading to numerical
//! instability.
//!
class LinearEquation : public Model, public ErrorFunction
{
public:
	//! Constructor
	LinearEquation(const Array<double>& mat, const Array<double>& vec);

	//! Destructor
	~LinearEquation();

	//! The model method throws an exception because
	//! this is not a data processing model.
	void model(const Array<double>& input, Array<double>& output);

	//! Computation of the error
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computation of the error and its derivative
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//! Return of the current solution
	void getSolution(Array<double>& solution);

protected:
	double error();
	double errorDerivative(Array<double>& derivative);

	const Array<double>& matrix;
	const Array<double>& vector;
};


#endif

