//===========================================================================
/*!
 *  \file KernelNearestNeighborRegression.h
 *
 *  \brief Kernel k-Nearest Neighbor Regression
 *
 *  \author  C. Igel
 *  \date    2010
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


#ifndef _KernelNearestNeighborRegression_H_
#define _KernelNearestNeighborRegression_H_

#include <ReClaM/Model.h>
#include <Array/ArrayOp.h>
#include <ReClaM/KernelNearestNeighbor.h>

//! The kernel nearest neighbor classifier is parameter free,
//! that is, it does not require training.
class KernelNearestNeighborRegression : public KernelNearestNeighbor
{
public:
	//! Constructor
	//!
	//! \param  kernelfunction  kernel function
	//! \param  k               number of neighbors to consider
	KernelNearestNeighborRegression(KernelFunction* kernelfunction, int k);

	//! Constructor
	//!
	//! \param  input           input patterns
	//! \param  target          input labels
	//! \param  kernelfunction  kernel function
	//! \param  k               number of neighbors to consider
	KernelNearestNeighborRegression(const Array<double>& input, const Array<double>& target, KernelFunction* kernelfunction, int k);

	//! Destructor
	~KernelNearestNeighborRegression();

	//! The model method does the regression
	void model(const Array<double>& input, Array<double>& output);

	//! Predict i-th pattern using the others, allows efficient LOO error computation
	void doRegression(int i, Array<double> &output);
protected:
	//! Do regression
	void doRegression(const Array<double> &pattern, Array<double> &output);
};


#endif


