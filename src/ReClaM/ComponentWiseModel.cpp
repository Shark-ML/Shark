//===========================================================================
/*!
*  \file ComponentWiseModel.cpp
*
*  \brief The ComponentWiseModel encapsulates the component wise application of a base model.
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

#include <ReClaM/ComponentWiseModel.h>


ComponentWiseModel::ComponentWiseModel(Model* pBase, int numberOfCopies)
{
	base = pBase;
	copies = numberOfCopies;

	inputDimension = numberOfCopies * pBase->getInputDimension();
	outputDimension = numberOfCopies * pBase->getOutputDimension();

	int p, pc = pBase->getParameterDimension();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) parameter(p) = pBase->getParameter(p);
}

ComponentWiseModel::~ComponentWiseModel()
{
}


void ComponentWiseModel::model(const Array<double>& input, Array<double> &output)
{
	int p, pc = getParameterDimension();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) base->setParameter(p, parameter(p));

	Array<double> in, out;
	int c;
	int i, ii, ic = base->getInputDimension();
	int o, oo, oc = base->getOutputDimension();

	if (input.ndim() == 1)
	{
		in.resize(ic, false);
		output.resize(copies * oc, false);
		ii = 0;
		oo = 0;
		for (c = 0; c < copies; c++)
		{
			for (i = 0; i < ic; i++, ii++)
			{
				in(i) = input(ii);
			}
			base->model(in, out);
			for (o = 0; o < oc; o++, oo++)
			{
				output(oo) = out(o);
			}
		}
	}
	else if (input.ndim() == 2)
	{
		int j, jc = input.dim(0);
		in.resize(jc, ic, false);
		output.resize(jc, copies * oc, false);
		ii = 0;
		oo = 0;
		for (c = 0; c < copies; c++)
		{
			for (i = 0; i < ic; i++, ii++)
			{
				for (j = 0; j < jc; j++) in(j, i) = input(j, ii);
			}
			base->model(in, out);
			for (o = 0; o < oc; o++, oo++)
			{
				for (j = 0; j < jc; j++) output(j, oo) = out(j, o);
			}
		}
	}
	else throw SHARKEXCEPTION("[ComponentWiseModel::model] Invalid number of dimensions.");
}

void ComponentWiseModel::modelDerivative(const Array<double>& input, Array<double>& derivative)
{
	int p, pc = getParameterDimension();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) base->setParameter(p, parameter(p));

	// TODO: this means some work not done right now ...
}

void ComponentWiseModel::modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative)
{
	int p, pc = getParameterDimension();
	parameter.resize(pc, false);
	for (p = 0; p < pc; p++) base->setParameter(p, parameter(p));

	// TODO: perform both model and modelDerivative at the same time ...
}

bool ComponentWiseModel::isFeasible()
{
	return base->isFeasible();
}

