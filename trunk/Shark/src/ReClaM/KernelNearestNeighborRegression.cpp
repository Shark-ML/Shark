//===========================================================================
/*!
 *  \file KernelNearestNeighborRegression.h
 *
 *  \brief Kernel k-Nearest Neighbor Regression
 *
 *  \author  T. Glasmachers, C. igel
 *  \date    2006, 2010
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


#include "KernelNearestNeighborRegression.h"

KernelNearestNeighborRegression::KernelNearestNeighborRegression(KernelFunction* kernelfunction, int k)
  : KernelNearestNeighbor(kernelfunction, k)
{
	inputDimension = outputDimension = 0;
}

KernelNearestNeighborRegression::KernelNearestNeighborRegression(const Array<double>& input, const Array<double>& target, KernelFunction* kernelfunction, int k)
: KernelNearestNeighbor(kernelfunction, k)
{}

KernelNearestNeighborRegression::~KernelNearestNeighborRegression()
{
}


void KernelNearestNeighborRegression::model(const Array<double>& input, Array<double>& output)
{
  if (bMustRecalc) Recalc();

	if (input.ndim() == 1)
	{
		output.resize(outputDimension, false);
		doRegression(input, output);
	}
	else if (input.ndim() == 2)
	{
		int j, s = input.dim(0);
		output.resize(s, outputDimension, false);
		Array<double> outputPattern(outputDimension);
		Array<double> inputPattern(inputDimension);
		for (j = 0; j < s; j++) {
		  inputPattern = input[j];
		  doRegression(inputPattern, outputPattern);
		  output[j] = outputPattern;
		}
	}
	else throw SHARKEXCEPTION("[KernelNearestNeighbor::model] invalid input dimension");
}

void KernelNearestNeighborRegression::doRegression(const Array<double> &pattern, Array<double> &output)
{
	int i, j, m, u, c, l = training_input.dim(0);
	double dist2, best;
	double norm2 = kernel->eval(pattern, pattern);
	std::vector<int> used;		// sorted list of neighbors
	for (i = 0; i < numberOfNeighbors; i++)
	{
		// find the nearest neightbound not already in the list
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

	output = 0;
	for (i = 0; i < numberOfNeighbors; i++) output += training_target[used[i]];
	output /= double(numberOfNeighbors);
}

void KernelNearestNeighborRegression::doRegression(const int index, Array<double> &output)
{
	Array<double> pattern = training_input[index];
	int i, j, m, u, c, l = training_input.dim(0);
	double dist2, best;
	double norm2 = kernel->eval(pattern, pattern);
	std::vector<int> used;		// sorted list of neighbors


	for (i = 0; i < numberOfNeighbors; i++)
	{
		// find the nearest neightbound not already in the list
		best = 1e100;
		m = 0;
		for (j = 0; j < i; j++)
		{
			if(m == index) continue;
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
			if(m == index) continue;
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

	output = 0;
	for (i = 0; i < numberOfNeighbors; i++) output += training_target[used[i]];
	output /= double(numberOfNeighbors);
}
