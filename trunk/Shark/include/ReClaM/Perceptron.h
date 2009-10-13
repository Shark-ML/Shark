//===========================================================================
/*!
*  \file Perceptron.h
*
*  \brief Perceptron online learning algorith
*
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


#ifndef _Perceptron_H_
#define _Perceptron_H_


#include <ReClaM/Optimizer.h>
#include <ReClaM/Svm.h>


//! \brief Perceptron online learning algorith
class Perceptron : public Optimizer
{
public:
	//! Constructor
	Perceptron();

	//! Destructor
	~Perceptron();

	void init(Model& model);
	void optimize(SVM& svm, const Array<double>& input, const Array<double>& target);
	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target);
};


#endif

