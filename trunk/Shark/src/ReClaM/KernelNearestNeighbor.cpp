//===========================================================================
/*!
 *  \file KernelNearestNeighbor.cpp
 *
 *  \brief Kernel k-Nearest Neighbor Classifier
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


#include <ReClaM/KernelNearestNeighbor.h>


KernelNearestNeighbor::KernelNearestNeighbor(KernelFunction* kernelfunction, int k)
: kernel(kernelfunction)
{
	numberOfNeighbors = k;

	int p, pc = kernelfunction->getParameterDimension();
	parameter.resize(pc + 1, false);
	parameter(0) = numberOfNeighbors;
	for (p = 0; p < pc; p++) parameter(p + 1) = kernelfunction->getParameter(p);
}

KernelNearestNeighbor::KernelNearestNeighbor(const Array<double>& input, const Array<double>& target, KernelFunction* kernelfunction, int k)
: kernel(kernelfunction)
{
	numberOfNeighbors = k;

	SetPoints(input, target);
}

KernelNearestNeighbor::~KernelNearestNeighbor()
{
}


//! Define a set of labeled points as a base for classification
void KernelNearestNeighbor::SetPoints(const Array<double>& input, const Array<double>& target)
{
	inputDimension = input.dim(1);
	outputDimension = target.dim(1);
	training_input = input;
	training_target = target;
	bMustRecalc = true;
}

void KernelNearestNeighbor::setParameter(unsigned int index, double value)
{
	if (index == 0)
	{
		numberOfNeighbors = (int)value;
		if (numberOfNeighbors < 1) numberOfNeighbors = 1;
		parameter(0) = numberOfNeighbors;
	}
	else
	{
		parameter(index) = value;
		kernel->setParameter(index-1, value);
	}
	bMustRecalc = true;
}

void KernelNearestNeighbor::model(const Array<double>& input, Array<double>& output)
{
	if (bMustRecalc) Recalc();

	if (input.ndim() == 1)
	{
		output.resize(1, false);
		output(0) = classify(input);
	}
	else if (input.ndim() == 2)
	{
		int j, s = input.dim(0);
		output.resize(s, 1, false);
		for (j = 0; j < s; j++) output(j, 0) = classify(input[j]);
	}
	else throw SHARKEXCEPTION("[KernelNearestNeighbor::model] invalid input dimension");
}

void KernelNearestNeighbor::Recalc()
{
	bMustRecalc = false;

	int i, l = training_input.dim(0);
	if (numberOfNeighbors > l) throw SHARKEXCEPTION("[KernelNearestNeighbor::Recalc] There are less input patterns than neighbors!");

	// compute the squared norms of the
	// training examples in the feature space
	diag.resize(l, false);
	for (i = 0; i < l; i++) diag(i) = kernel->eval(training_input[i], training_input[i]);
}

double KernelNearestNeighbor::classify(Array<double> pattern)
{
	int i, j, m, u, c, l = training_input.dim(0);
	double dist2, best;
	double norm2 = kernel->eval(pattern, pattern);
	std::vector<int> used;		// sorted list of neighbors
	for (i = 0; i < numberOfNeighbors; i++)
	{
		// find the nearest neighbor not already in the list
		best = 1e100;
		m = 0;
		for (j = 0; j < i; j++)
		{
			u = used[j];
			for (; m < u; m++)
			{
				dist2 = diag(m) + norm2 - 2.0 * kernel->eval(training_input[m], pattern);
				if (dist2 < best)
				{
					best = dist2;
					c = m;
				}
			}
			m++;
		}
		for (; m < l; m++)
		{
			dist2 = diag(m) + norm2 - 2.0 * kernel->eval(training_input[m], pattern);
			if (dist2 < best)
			{
				best = dist2;
				c = m;
			}
		}

		// insert the nearest neighbor into the sorted list
		for (j = 0; j < i; j++) if (used[j] >= c) break;
		if (j == i) used.push_back(c);
		else used.insert(used.begin() + j, c);
	}
	double mean = 0.0;
	for (i = 0; i < numberOfNeighbors; i++) mean += training_target(used[i], 0);
	return (mean > 0.0) ? 1.0 : -1.0;
}
