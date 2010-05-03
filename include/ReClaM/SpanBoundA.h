//===========================================================================
/*!
*  \file SpanBoundA.h
*
*  \brief approximate SpanBound for the 1-norm SVM
*
*  \author  T. Glasmachers
*  \date    2010
*
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


#ifndef _SpanBoundA_H_
#define _SpanBoundA_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>


//! \brief approximate SpanBound for the 1-norm SVM
class SpanBoundA : public ErrorFunction
{
public:
	//! Constructor
	SpanBoundA();

	//! Destructor
	~SpanBoundA();


	//! Computation of the span bound
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computation of the span bound and its derivative
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//! set the maximum number of iterations
	//! for the quadratic program solver
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	//! return the number of iterations last
	//! used by the quadratic program solver
	inline SharkInt64 iterations()
	{
		return iter;
	}

protected:
	//! maximum number of #C_Solver iterations
	SharkInt64 maxIter;

	//! last number if #C_Solver iterations
	SharkInt64 iter;
};


#endif
