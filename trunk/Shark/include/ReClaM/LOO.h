//===========================================================================
/*!
*  \file LOO.h
*
*  \author  T. Glasmachers
*  \date    2006
*
*  \brief Leave One Out (LOO) Error for Support Vector Machines
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
*  This file is part of Shark. This library is free software;
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

#ifndef _LOO_H_
#define _LOO_H_


#include <SharkDefs.h>
#include <ReClaM/Model.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/QuadraticProgram.h>


/*!
 *  \brief Leave One Out (LOO) Error for Support Vector Machines
 *
 *
 *  The leave one out error is as extreme variant of the
 *  cross validation error, where all examples but one are
 *  used for training and only the left out example is used
 *  for testing. For SVM classifiers, the LOO error can be
 *  computed easily as it is known that non-support-vectors
 *  do not contribute. On the other hand, for support
 *  vectors the optimization can be started from a feasible
 *  solution near the full SVM classifier realizing fast
 *  convergence.
 */
class LOO : public ErrorFunction
{
public:
	//! Constructor
	LOO();

	//! Destructor
	~LOO();


	//! Compute the leave one out error for the C-SVM model.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Compute the SVM offset b.
	double ComputeB(QpSvmDecomp& qp, const Array<double>& lower, const Array<double>& upper, const Array<double>& alpha);

	//! set the maximum number of iterations
	//! for the quadratic program solver
	void setMaxIterations(SharkInt64 maxiter = -1);

	//! return the number of iterations last
	//! used by the quadratic program solver
	inline SharkInt64 iterations()
	{
		return iter;
	}

protected:
	//! maximum number of #QpSvmDecomp iterations
	SharkInt64 maxIter;

	//! last number if #QpSvmDecomp iterations
	SharkInt64 iter;
};


#endif

