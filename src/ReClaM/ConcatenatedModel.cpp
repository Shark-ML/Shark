//===========================================================================
/*!
*  \file ConcatenatedModel.cpp
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


#include <ReClaM/ConcatenatedModel.h>


ConcatenatedModel::ConcatenatedModel()
{
}

ConcatenatedModel::~ConcatenatedModel()
{
	int i, ic = models.size();
	for (i = 0; i < ic; i++) delete models[i];
	models.clear();
}


void ConcatenatedModel::AppendModel(Model* pModel)
{
	if (models.size() > 0)
	{
		Model* pLast = models[models.size() - 1];
		if (pLast->getOutputDimension() != pModel->getInputDimension())
		{
			throw SHARKEXCEPTION("[ConcatenatedModel::AppendModel] Dimension conflict");
		}

		int d = parameter.dim(0);
		int i, ic = pModel->getParameterDimension();
		parameter.resize(d + ic, true);
		for (i = 0; i < ic; i++) parameter(d + i) = pModel->getParameter(i);
	}
	else
	{
		int i, ic = pModel->getParameterDimension();
		parameter.resize(ic, false);
		for (i = 0; i < ic; i++) parameter(i) = pModel->getParameter(i);
		inputDimension = pModel->getInputDimension();
	}
	outputDimension = pModel->getOutputDimension();

	models.push_back(pModel);
}

void ConcatenatedModel::model(const Array<double>& input, Array<double> &output)
{
	Array<double> temp[2];
	int i, ic = models.size();
	int t = 0;
	int p, pc, pp = 0;

	// copy the parameters into the elementary models
	for (i = 0; i < ic; i++)
	{
		pc = models[i]->getParameterDimension();
		for (p = 0; p < pc; p++)
		{
			models[i]->setParameter(p, parameter(pp));
			pp++;
		}
	}

	// propagate the input through the model chain
	for (i = 0; i < ic; i++)
	{
		if (i == 0)
		{
			models[i]->model(input, temp[0]);
		}
		else if (i == ic - 1)
		{
			models[i]->model(temp[t], output);
		}
		else
		{
			models[i]->model(temp[t], temp[1 - t]);
			t = 1 - t;
		}
	}
}

void ConcatenatedModel::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	int i, ic = models.size();
	int p, pc, pp = 0;

	// copy the parameters into the elementary models
	for (i = 0; i < ic; i++)
	{
		pc = models[i]->getParameterDimension();
		for (p = 0; p < pc; p++)
		{
			models[i]->setParameter(p, parameter(pp));
			pp++;
		}
	}

	// TODO: this means some work not done right now ...
}

void ConcatenatedModel::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	int i, ic = models.size();
	int p, pc, pp = 0;

	// copy the parameters into the elementary models
	for (i = 0; i < ic; i++)
	{
		pc = models[i]->getParameterDimension();
		for (p = 0; p < pc; p++)
		{
			models[i]->setParameter(p, parameter(pp));
			pp++;
		}
	}

	// TODO: perform both model and modelDerivative at the same time ...
}

bool ConcatenatedModel::isFeasible()
{
	int i, ic = models.size();
	for (i=0; i<ic; i++) if (! models[i]->isFeasible()) return false;
	return true;
}

