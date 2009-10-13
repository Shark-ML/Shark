//===========================================================================
/*!
 *  \file LDA.h
 *
 *  \brief Linear Discriminant Analysis (LDA)
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par
 *      This implementation is based upon a class removed from
 *      the LinAlg package, written by M. Kreutz in 1998.
 *
 *  \par Copyright (c) 2007:
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


#ifndef _LDA_H_
#define _LDA_H_


#include <ReClaM/LinearModel.h>
#include <ReClaM/Optimizer.h>


//!
//! \brief Linear Discriminant Analysis (LDA)
//!
//! \par
//! This optimizer trains a linear model using linear discriminant analysis.
//!
class LDA : public Optimizer
{
public:
	//! Constructor
	LDA();

	//! Destructor
	~LDA();

	//! The initialization checks for compatibility of the model.
	//! It must be either a LinearClassifier or a AffineLinearFunction
	//! in case of binary classification.
	void init(Model& model);

	//! The inherited optimize function inherits one of the specialized
	//! versions for LinearClassifier or AffineLinearFunction models.
	//! ATTENTION: The target data must be organized differently for
	//! these models!
	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target);

	//! Compute the LDA solution for a multi-class problem. Let c be the
	//! number of classes. Then the targets are assumes to be vectors in
	//! \f$ R^c \f$ padded with zeros and a single one in the component
	//! corresponding to the target class.
	double optimize(LinearClassifier& model, const Array<double>& input, const Array<double>& target);

	//! Compute the LDA solution for a binary classification problem with
	//! is stated as a zero-thresholded affine linear function.
	//! In this case the targets must be one-dimensional, taking the values
	//! +1 and -1.
	double optimize(AffineLinearFunction& model, const Array<double>& input, const Array<double>& target);
};


#endif

