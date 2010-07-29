//===========================================================================
/*!
 *  \file NegativeLogLikelihood.cpp
 *
 *  \brief negative logarithm of the likelihood of a probabilistic binary classification model
 *
 *  \author  T. Glasmachers
 *  \date    2008, 2010
 *
 *  \par Copyright (c) 1999-2010:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#include <ReClaM/NegativeLogLikelihood.h>
#include <Array/ArrayOp.h>


NegativeLogLikelihood::NegativeLogLikelihood(NegativeLogLikelihood::eLabelType labelType, unsigned int classes)
{
	m_LabelType = labelType;
	m_classes = classes;
	if (m_LabelType == elBinary && classes != 2) throw SHARKEXCEPTION("[NegativeLogLikelihood::NegativeLogLikelihood] invalid initialization");
}

NegativeLogLikelihood::~NegativeLogLikelihood()
{
}


double NegativeLogLikelihood::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	double ret = 0.0;

	if (m_LabelType == elBinary)
	{
		if (input.ndim() == 1)
		{
			Array<double> output(1);
			model.model(input, output);
			if (target(0) > 0.0) ret -= log(output(0));
			else ret -= log(1.0 - output(0));
		}
		else if (input.ndim() == 2)
		{
			int i, ic = input.dim(0);
			Array<double> output(ic, 1);
			model.model(input, output);
			for (i=0; i<ic; i++)
			{
				if (target(i, 0) > 0.0) ret -= log(output(i, 0));
				else ret -= log(1.0 - output(i, 0));
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::error] invalid dimension");
	}
	else if (m_LabelType == elNumber)
	{
		if (input.ndim() == 1)
		{
			Array<double> output(1);
			model.model(input, output);
			unsigned int t = (unsigned int)target(0);
			RANGE_CHECK(t < m_classes);
			ret -= log(output(t));
		}
		else if (input.ndim() == 2)
		{
			int i, ic = input.dim(0);
			Array<double> output(ic, 1);
			model.model(input, output);
			for (i=0; i<ic; i++)
			{
				unsigned int t = (unsigned int)target(i, 0);
				RANGE_CHECK(t < m_classes);
				ret -= log(output(i, t));
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::error] invalid dimension");
	}
	else if (m_LabelType == elVector)
	{
		if (input.ndim() == 1)
		{
			Array<double> output(m_classes);
			model.model(input, output);
			ret -= log(scalarProduct(output, target));
		}
		else if (input.ndim() == 2)
		{
			int i, ic = input.dim(0);
			Array<double> output(m_classes);
			for (i=0; i<ic; i++)
			{
				model.model(input[i], output);
				ret -= log(scalarProduct(output, target[i]));
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::error] invalid dimension");
	}

	return ret;
}

double NegativeLogLikelihood::errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative)
{
	double ret = 0.0;

	if (m_LabelType == elBinary)
	{
		int p, pc = model.getParameterDimension();
		derivative.resize(pc, false);
		derivative = 0.0;
		Array<double> output(1);
		Array<double> dmdw;
		if (input.ndim() == 1)
		{
			SIZE_CHECK(target.dim(0) == 1);
			model.modelDerivative(input, output, dmdw);
			double o = output(0);
			if (target(0) > 0.0)
			{
				ret -= log(o);
			}
			else
			{
				ret -= log(1.0 - o);
				o -= 1.0;
			}
			for (p=0; p<pc; p++) derivative(p) -= dmdw(0, p) / o;
		}
		else if (input.ndim() == 2)
		{
			SIZE_CHECK(target.dim(1) == 1);
			Array<double> der(pc);
			int i, ic = input.dim(0);
			for (i=0; i<ic; i++)
			{
				model.modelDerivative(input[i], output, dmdw);
				double o = output(0);
				if (target(i, 0) > 0.0)
				{
					ret -= log(o);
				}
				else
				{
					ret -= log(1.0 - o);
					o -= 1.0;
				}
				for (p=0; p<pc; p++) derivative(p) -= dmdw(0, p) / o;
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::errorDerivative] invalid dimension");
	}
	else if (m_LabelType == elNumber)
	{
		int p, pc = model.getParameterDimension();
		derivative.resize(pc, false);
		derivative = 0.0;
		Array<double> output(1);
		Array<double> dmdw;
		if (input.ndim() == 1)
		{
			SIZE_CHECK(target.dim(0) == 1);
			model.modelDerivative(input, output, dmdw);
			unsigned int t = (unsigned int)target(0);
			RANGE_CHECK(t < m_classes);
			double o = output(t);
			ret -= log(o);
			for (p=0; p<pc; p++)
				derivative(p) -= dmdw(t, p) / o;
		}
		else if (input.ndim() == 2)
		{
			SIZE_CHECK(target.dim(1) == 1);
			Array<double> der(pc);
			int i, ic = input.dim(0);
			for (i=0; i<ic; i++)
			{
				model.modelDerivative(input[i], output, dmdw);
				unsigned int t = (unsigned int)target(i, 0);
				RANGE_CHECK(t < m_classes);
				double o = output(t);
				ret -= log(o);
				for (p=0; p<pc; p++)
					derivative(p) -= dmdw(t, p) / o;
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::errorDerivative] invalid dimension");
	}
	else if (m_LabelType == elVector)
	{
		int p, pc = model.getParameterDimension();
		derivative.resize(pc, false);
		derivative = 0.0;
		Array<double> output(m_classes);
		Array<double> dmdw;
		if (input.ndim() == 1)
		{
			SIZE_CHECK(target.dim(0) == m_classes);
			model.modelDerivative(input, output, dmdw);
			unsigned int t;
			for (t=0; t<m_classes; t++) if (target(t) > 0.0) break;
			RANGE_CHECK(t < m_classes);
			double o = output(t);
			ret -= log(o);
			for (p=0; p<pc; p++)
				derivative(p) -= dmdw(t, p) / o;
		}
		else if (input.ndim() == 2)
		{
			SIZE_CHECK(target.dim(1) == m_classes);
			Array<double> der(pc);
			int i, ic = input.dim(0);
			for (i=0; i<ic; i++)
			{
				model.modelDerivative(input[i], output, dmdw);
				unsigned int t;
				for (t=0; t<m_classes; t++) if (target(i, t) > 0.0) break;
				RANGE_CHECK(t < m_classes);
				double o = output(t);
				ret -= log(o);
				for (p=0; p<pc; p++)
					derivative(p) -= dmdw(t, p) / o;
			}
		}
		else throw SHARKEXCEPTION("[NegativeLogLikelihood::errorDerivative] invalid dimension");
	}

	return ret;
}
