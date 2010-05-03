//===========================================================================
/*!
*  \file SpanBound2.h
*
*  \brief SpanBound for the 2-norm SVM
*
*  \author  T. Glasmachers
*  \date    2006
*
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


#ifndef _SpanBound2_H_
#define _SpanBound2_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>


//! \brief SpanBound for the 2-norm SVM
class SpanBound2 : public ErrorFunction
{
public:
	//! Constructor
	SpanBound2(bool verbose = false);

	//! Destructor
	~SpanBound2();


	//! Computation of the span bound
	//! If the quadratic program solver does not reach
	//! the optimum within the maximal number of iterations
	//! the method returns 1e100.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

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
	//! output the status to stdout?
	bool verbose;

	//! maximum number of #C_Solver iterations
	SharkInt64 maxIter;

	//! last number if #C_Solver iterations
	SharkInt64 iter;
};


#endif
