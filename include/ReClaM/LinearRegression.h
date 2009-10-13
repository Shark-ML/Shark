//===========================================================================
/*!
 *  \file LinearRegression.h
 *
 *  \brief Linear Regression
 *
 *  \par
 *      This implementation is based on a class removed from the
 *      LinAlg package, written by M. Kreutz in 1998.
 *
 *  \author T. Glasmachers
 *  \date 2007
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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


#ifndef _LinearRegression_H_
#define _LinearRegression_H_


#include <ReClaM/LinearModel.h>
#include <ReClaM/Optimizer.h>


//===========================================================================
/*!
 *  \brief Linear Regression
 *
 *  Linear Regression builds an affine linear model
 *  \f$ f(x) = A x + b \f$ minimizing the squared
 *  error from a dataset of pairs of vectors (x, y).
 *  That is, the error
 *  \f$ \sum_i (f(x_i) - y_i)^2 \f$ is minimized.
 *  The solution to this problem is found analytically.
 */
class LinearRegression : public Optimizer
{
public:
	LinearRegression();
	~LinearRegression();

	void init(Model& model);

	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target);
	double optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& target);
};


#endif

