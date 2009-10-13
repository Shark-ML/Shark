//===========================================================================
/*!
*  \file Perceptron.cpp
*
*  \brief Perceptron online learning algorith,
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


#include <ReClaM/Perceptron.h>


Perceptron::Perceptron()
{
}

Perceptron::~Perceptron()
{
}


void Perceptron::init(Model& model)
{
	SVM* svm = dynamic_cast<SVM*>(&model);
	if (svm == NULL) throw SHARKEXCEPTION("[Perceptron::init] model is not a valid SVM object");
}

void Perceptron::optimize(SVM& svm, const Array<double>& input, const Array<double>& target)
{
	int i, ic = input.dim(0);
	bool err;
	int iter = 0;

	svm.SetTrainingData(input);
	for (i = 0; i <= ic; i++) svm.setParameter(i, 0.0);

	do
	{
		err = false;
		for (i = 0; i < ic; i++)
		{
			if (svm.model(input[i]) * target(i, 0) <= 0.0)
			{
				svm.setParameter(i, svm.getParameter(i) + target(i, 0));
				err = true;
			}
		}
		if (iter > 10000 * ic) break;	// probably non-separable data
		iter++;
	}
	while (err);
}

double Perceptron::optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
{
	SVM* svm = dynamic_cast<SVM*>(&model);
	if (svm == NULL) throw SHARKEXCEPTION("[Perceptron::optimize] model is not a valid SVM object");

	optimize(*svm, input, target);
	return 0.0;
}

