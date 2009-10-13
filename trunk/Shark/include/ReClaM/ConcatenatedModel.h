//===========================================================================
/*!
*  \file ConcatenatedModel.h
*
*  \brief The ConcatenatedModel encapsulates a chain of basic models.
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


#ifndef _ConcatenatedModel_H_
#define _ConcatenatedModel_H_


#include <ReClaM/Model.h>
#include <vector>


//!
//! \brief The ConcatenatedModel encapsulates a chain of basic models.
//!
//! \par
//! This class allows for the concatenation of models
//! \f$ f_{\alpha} : R^n \rightarrow R^m \f$ and \f$ g_{\beta} : R^m \rightarrow R^k \f$
//! in the form h_{\alpha, \beta} = f_{\alpha} \circ g_{\beta} : R^n \rightarrow R^k \f$.
//! That is, for prediction several models are applied to the data as a chain
//! or pipeline. The parameters of the joint model are composed of the parameters
//! of all base models.
//!
class ConcatenatedModel : public Model
{
public:
	//! Constructor
	ConcatenatedModel();

	//! Destructor
	~ConcatenatedModel();


	//! Append a model to the chain.
	//! Note that a once added model becomes part of the ComcatenatedModel
	//! object, that is the ComcatenatedModel object destroys the added
	//! model in its destructor.
	void AppendModel(Model* pModel);

	//! Evaluate the concatenated model
	void model(const Array<double>& input, Array<double> &output);

	//! We will have to implement the chain rule for all cases...
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! We will have to implement the chain rule for all cases...
	void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);

	//! Check whether the parameters define a valid model
	bool isFeasible();

protected:
	//! chain of models
	std::vector<Model*> models;
};


#endif

