//===========================================================================
/*!
 *  \file NegativeLogLikelihood.h
 *
 *  \brief negative logarithm of the likelihood of a probabilistic binary classification model
 *
 *  \author  T. Glasmachers
 *  \date    2008
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


#ifndef _NegativeLogLikelihood_H_
#define _NegativeLogLikelihood_H_


#include <ReClaM/ErrorFunction.h>


//!
//! \brief negative logarithm of the likelihood of a probabilistic binary classification model
//!
//! \par
//! This class makes two assumptions:
//! First, the data must be a binary classification problem
//! using labels +1 and -1. Second, the model must output
//! values in the unit interval [0, 1] which are interpreted
//! as the probability of an example being positive.
//!
//! \par
//! With data \f$ x_i, y_i \f$, a model f the error function is
//! \f$ -\sum_{y \in Y} \sum_{y_i = j} \log(f_j(x_i)) \f$.
//!
class NegativeLogLikelihood : public ErrorFunction
{
public:
	enum eLabelType
	{
		elBinary,			//! -1 or +1
		elNumber,			//! 0, ..., #classes-1
		elVector,			//! n-th unit vector
	};

	//! Constructor
	NegativeLogLikelihood(eLabelType labelType = elBinary, unsigned int classes = 2);

	//! Destructor
	~NegativeLogLikelihood();


	//! error computation, see class description
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! error computation with derivatives, see class description
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

protected:
	//! type of label information expected
	eLabelType m_LabelType;

	//! number of classes spanning the probability simplex
	unsigned int m_classes;
};


#endif
