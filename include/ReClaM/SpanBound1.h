//===========================================================================
/*!
*  \file SpanBound1.h
*
*  \brief SpanBound for the 1-norm SVM
*
*  \author  T. Glasmachers
*  \date    2008
*
*
*  \par Copyright (c) 1999-2008:
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


#ifndef _SpanBound1_H_
#define _SpanBound1_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/Svm.h>


//! \brief SpanBound for the 1-norm SVM
class SpanBound1 : public ErrorFunction
{
public:
	//! Constructor
	SpanBound1(bool verbose = false);

	//! Destructor
	~SpanBound1();


	//! This method computes an upper bound on the
	//! leave one out error.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! This method computes an upper bound on the
	//! leave one out error as well as its derivative
	//! w.r.t. the kernel parameters and C.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//! set the maximum number of iterations
	//! for the quadratic program solver
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxiter = maxiter;
	}

protected:
	double bound(C_SVM* csvm, int p, const Array<double>& input, const Array<double>& target);
	double boundDerivative(C_SVM* csvm, int p, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

	//! output the status to stdout?
	bool verbose;

	//! maximum number of iterations for the quadratic program solver
	SharkInt64 maxiter;
};


#endif
