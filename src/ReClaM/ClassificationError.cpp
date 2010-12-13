//===========================================================================
/*!
*  \file ClassificationError.cpp
*
*  \brief Compute the fraction of classification errors
*
*  \author  T. Glasmachers
*  \date    2006
*
*
*  \par
*      Both the #ClassificationError and the #BalancedClassificationError
*      assume one dimensional model output and target values. There are
*      two variants, that is, the output and target arrays may be one
*      dimensional with the dimension representing the data points, or
*      the arrays may contain an exclicit second dimension. In the second
*      case the output dimension must equal 1. In all cases there has to
*      be one real valued output and target per input pattern.
*      This covers both the cases of support vector machines and neural
*      networks with a single output neuron. For multi class neural
*      networks refer the the winner takes all class.
*
*  \par
*      It is assumed that the classication is done by comparison with a
*      threshold, usually zero. This includes the special case of taking
*      the sign function for classification.
*      For neural networks with sigmoidal transfer function of the output
*      neuron it is usually assumed that the classification boundary is
*      the value \f$ 0.5 \f$. Therefore it is possible to provide a
*      nonzero threshold parameter at construction time.
*
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


#include <ReClaM/ClassificationError.h>
#include <Array/ArrayOp.h>


ClassificationError::ClassificationError(double threshold)
{
	this->threshold = threshold;
}

ClassificationError::~ClassificationError()
{
}


double ClassificationError::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	if (input.ndim() != 2) throw SHARKEXCEPTION("[ClassificationError::error] invalid number of input dimensions.");
	if (target.ndim() != 2) throw SHARKEXCEPTION("[ClassificationError::error] invalid number of target dimensions.");
	unsigned int fp = 0;
	unsigned int fn = 0;
	unsigned int pos = 0;
	unsigned int neg = 0;
	unsigned int i, ic = input.dim(0);
	SIZE_CHECK(target.dim(0) == ic);
	Array<double> output;
	model.model(input, output);

	if (target.dim(1) != 1 || output.ndim() != 2 || output.dim(1) != 1) throw SHARKEXCEPTION("[ClassificationError::error] target and output dimension are incompatible.");

	for (i = 0; i < ic; i++)
	{
		if (target(i, 0) > threshold)
		{
			if (output(i, 0) <= threshold) fn++;
			pos++;
		}
		else
		{
			if (output(i, 0) > threshold) fp++;
			neg++;
		}
	}

	// remember false positive and false negative rate
	// for later reference
	fpr = (double)fp / (double)neg;
	fnr = (double)fn / (double)pos;

	return ((double)(fp + fn)) / ic;
}


////////////////////////////////////////////////////////////


BalancedClassificationError::BalancedClassificationError(double threshold)
{
	this->threshold = threshold;
}

BalancedClassificationError::~BalancedClassificationError()
{
}


double BalancedClassificationError::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	if (input.ndim() != 2) throw SHARKEXCEPTION("[BalancedClassificationError::error] invalid number of input dimensions.");
	if (target.ndim() != 2) throw SHARKEXCEPTION("[BalancedClassificationError::error] invalid number of target dimensions.");
	unsigned int p = 0;
	unsigned int n = 0;
	unsigned int a = 0;
	unsigned int b = 0;
	unsigned int i, ic = input.dim(0);
	SIZE_CHECK(target.dim(0) == ic);
	Array<double> output;
	model.model(input, output);

	if (target.dim(1) != 1 || target.dim(1) != output.dim(1)) throw SHARKEXCEPTION("[BalancedClassificationError::error] target and output dimension are incompatible.");

	for (i = 0; i < ic; i++)
	{
		if (target(i, 0) > threshold)
		{
			p++;
			if (output(i, 0) <= threshold) a++;
		}
		else
		{
			n++;
			if (output(i, 0) > threshold) b++;
		}
	}

	return 0.5 *((((double)a) / p) + (((double)b) / n));
}


////////////////////////////////////////////////////////////


ZeroOneLoss::ZeroOneLoss()
{
}

ZeroOneLoss::~ZeroOneLoss()
{
}


double ZeroOneLoss::error(Model& model, const Array<double>& input, const Array<double>& target)
{
	if (input.ndim() != 2) throw SHARKEXCEPTION("[ZeroOneLoss::error] invalid number of input dimensions.");
	if (target.ndim() != 2) throw SHARKEXCEPTION("[ZeroOneLoss::error] invalid number of target dimensions.");
	unsigned int i, ic = input.dim(0);
	SIZE_CHECK(target.dim(0) == ic);
	Array<double> output;
	model.model(input, output);

	if (target.dim(1) != output.dim(1)) throw SHARKEXCEPTION("[ZeroOneLoss::error] target and output dimensions are incompatible.");

	unsigned int err = 0;
	for (i = 0; i < ic; i++)
	{
		if (output[i] != target[i]) err++;
	}

	return ((double)err / (double)ic);
}
