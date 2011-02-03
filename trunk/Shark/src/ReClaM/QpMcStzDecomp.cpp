//===========================================================================
/*!
 *  \file QpMcStzDecomp.cpp
 *
 *  \brief Quadratic programming for Multi-Class Support Vector Machines with Sum-To-Zero constraint
 *
 *  \author  T. Glasmachers
 *  \date	2010
 *
 *  \par Copyright (c) 1999-2010:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <Array/ArrayIo.h>
#include <Array/ArrayOp.h>
#include <LinAlg/LinAlg.h>
#include <ReClaM/QuadraticProgram.h>

#include <math.h>
#include <iostream>
#include <iomanip>


using namespace std;


// useful exchange macros for Array<T> and std::vector<T>
#define XCHG_A(t, a, i, j) {t temp; temp = a(i); a(i) = a(j); a(j) = temp;}
#define XCHG_V(t, a, i, j) {t temp; temp = a[i]; a[i] = a[j]; a[j] = temp;}


////////////////////////////////////////////////////////////


QpMcStzDecomp::QpMcStzDecomp(CachedMatrix& kernel)
: kernelMatrix(kernel)
{
	examples = kernelMatrix.getMatrixSize();
	maxIter = -1;
}

QpMcStzDecomp::~QpMcStzDecomp()
{
}


void QpMcStzDecomp::Solve(unsigned int classes,
						const double* modifiers,
						const Array<double>& target,
						const Array<double>& linearPart,
						const Array<double>& lower,
						const Array<double>& upper,
						Array<double>& solutionVector,
						double eps)
{
	SIZE_CHECK(target.ndim() == 2);
	SIZE_CHECK(linearPart.ndim() == 1);
	SIZE_CHECK(lower.ndim() == 1);
	SIZE_CHECK(upper.ndim() == 1);
	SIZE_CHECK(solutionVector.ndim() == 1);

	this->classes = classes;
	variables = examples * classes;
	memcpy(m_modifier, modifiers, sizeof(m_modifier));

	SIZE_CHECK(target.dim(0) == examples);
	SIZE_CHECK(target.dim(1) == 1);
	SIZE_CHECK(linearPart.dim(0) == variables);
	SIZE_CHECK(lower.dim(0) == variables);
	SIZE_CHECK(upper.dim(0) == variables);
	SIZE_CHECK(solutionVector.dim(0) == variables);

	unsigned int a, b, bc, i, j;
	float* q = NULL;

	// prepare lists
	alpha.resize(variables, false);
	gradient.resize(variables, false);
	linear.resize(variables, false);
	boxMin.resize(variables, false);
	boxMax.resize(variables, false);
	example.resize(examples);
	variable.resize(variables);
	storage.resize(variables);

	// prepare list contents
	for (i = 0; i < examples; i++)
	{
		example[i].index = i;
		example[i].label = (unsigned int)target(i, 0);
		example[i].active = classes;
		example[i].variables = &storage[classes * i];
		example[i].diagonal = kernelMatrix.Entry(i, i);
	}
	for (i = 0; i < variables; i++)
	{
		unsigned int e = i / classes;
		unsigned int y = example[e].label;
		unsigned int v = i % classes;
		variable[i].example = e;
		variable[i].index = v;
		variable[i].label = v;
		variable[i].diagonal = Modifier(y, y, v, v) * kernelMatrix.Entry(e, e);
		storage[i] = i;
	}

	// prepare the solver internal variables
	boxMin = lower;
	boxMax = upper;
	linear = linearPart;
	gradient = linearPart;
	alpha = solutionVector;

	epsilon = eps;

	activeEx = examples;
	activeVar = variables;

	unsigned int e = 0xffffffff;
	for (i = 0; i < variables; i++)
	{
		if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpMcStzDecomp::Solve] The feasible region is empty.");

		double v = alpha(i);
		if (v != 0.0)
		{
			unsigned int ei = variable[i].example;
			unsigned int vi = variable[i].label;
			unsigned int yi = example[ei].label;
			if (ei != e)
			{
				q = kernelMatrix.Row(ei, 0, activeEx);
				e = ei;
			}

			for (a = 0; a < activeEx; a++)
			{
				double k = q[a];
				tExample* ex = &example[a];
				unsigned int yj = ex->label;
				bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					j = ex->variables[b];
					unsigned int vj = variable[j].label;
					double km = Modifier(yi, yj, vi, vj);
					gradient(j) -= v * km * k;
				}
			}
		}
	}

	bUnshrinked = false;
	unsigned int shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

	// initial shrinking (useful for dummy variables and warm starts)
	Shrink();

	// decomposition loop
	iter = 0;
	optimal = false;
	while (iter != maxIter)
	{
		// select a working set and check for optimality
		if (SelectWorkingSet(i, j))
		{
			// seems to be optimal

			// do costly unshrinking
			Unshrink(true);

			// check again on the whole problem
			if (SelectWorkingSet(i, j))
			{
				optimal = true;
				break;
			}

			// shrink again
			Shrink();
			shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

			SelectWorkingSet(i, j);
		}

		// update
		{
			double ai = alpha(i);
			double aj = alpha(j);
// 			double Li = boxMin(i);
			double Lj = boxMin(j);
			double Ui = boxMax(i);
// 			double Uj = boxMax(j);
			unsigned int eij = variable[i].example;
			ASSERT(variable[j].example == eij);
			unsigned int vi = variable[i].label;
			unsigned int vj = variable[j].label;
			unsigned int yij = example[eij].label;

			// get the matrix row corresponding to the working set
			q = kernelMatrix.Row(eij, 0, activeEx);

			// update alpha, that is, solve the sub-problem defined by i and j
			double numerator = gradient(i) - gradient(j);
			double denominator = variable[i].diagonal + variable[j].diagonal
					- 2.0 * Modifier(yij, yij, vi, vj) * q[eij];
			double mu = numerator / denominator;

			bool i_at_bound = false;
			bool j_at_bound = false;
			if (ai + mu > Ui) { i_at_bound = true; mu = Ui - ai; }
			if (aj - mu < Lj) { j_at_bound = true; mu = aj - Lj; }
			if (i_at_bound) alpha(i) = Ui; else alpha(i) += mu;
			if (j_at_bound) alpha(j) = Lj; else alpha(j) -= mu;

			// update the gradient
			for (a = 0; a < activeEx; a++)
			{
				double k = q[a];
				tExample* ex = &example[a];
				unsigned int yc = ex->label;
				bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					unsigned int c = ex->variables[b];
					unsigned int vc = variable[c].label;
					double km_i = Modifier(yij, yc, vi, vc);
					double km_j = Modifier(yij, yc, vj, vc);
					gradient(c) -= mu * (km_i - km_j) * k;
				}
			}
		}

		shrinkCounter--;
		if (shrinkCounter == 0)
		{
			// shrink the problem
			Shrink();

			shrinkCounter = (activeVar < 1000) ? activeVar : 1000;
		}

		iter++;
	}

	if (iter == maxIter) optimal = false;

	// return alpha
	for (i = 0; i < variables; i++)
	{
		unsigned int j = classes * example[variable[i].example].index + variable[i].label;
		solutionVector(j) = alpha(i);
	}
}

bool QpMcStzDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	unsigned int e;
	double worst_gap = 0.0;
	double best_gain = 0.0;

	// loop over the examples, defining groups of variables
	// among which we do second-order working set selection
	for (e=0; e<activeEx; e++)
	{
		tExample* ex = &example[e];
		double k = ex->diagonal;
		unsigned int y = ex->label;
		unsigned int b, bc = ex->active;

		// select first variable according to maximal gradient
		unsigned int ii = 0x7fffffff;
		double l_u = -1e100;
		double l_d = 1e100;
		for (b=0; b<bc; b++)
		{
			unsigned int c = ex->variables[b];
			double v = alpha(c);
			double g = gradient(c);
			if (v < boxMax(c))
			{
				if (g > l_u)
				{
					l_u = g;
					ii = c;
				}
			}
			if (v > boxMin(c))
			{
				if (g < l_d)
				{
					l_d = g;
					ii = c;
				}
			}
		}
		if (ii == 0x7fffffff) continue;
		double gap = l_u - l_d;
		if (gap > worst_gap) worst_gap = gap;

		// select second variable according to maximal unconstrained gain
		unsigned int vi = variable[ii].label;
		double aii = alpha(ii);
		double gii = gradient(ii);
		double mii = Modifier(y, y, vi, vi);
		for (b=0; b<bc; b++)
		{
			unsigned int c = ex->variables[b];
			if (c == ii) continue;
			double v = alpha(c);
			double g = gradient(c);
			double numerator = gii - g;
			if (numerator == 0.0) continue;
			if (numerator > 0.0 && (aii == boxMax(ii) || v == boxMin(c))) continue;
			if (numerator < 0.0 && (aii == boxMin(ii) || v == boxMax(c))) continue;

			unsigned int vc = variable[c].label;
			double denominator = (mii + Modifier(y, y, vc, vc) - 2.0 * Modifier(y, y, vi, vc)) * k;
			double gain = numerator * numerator / denominator;
			if (gain > best_gain)
			{
				i = ii;
				j = c;
				best_gain = gain;
			}
		}
	}

	if (gradient(i) < gradient(j)) std::swap(i, j);

	// return KKT stopping condition
	return (worst_gap < epsilon);
}

void QpMcStzDecomp::Shrink()
{
	int a, e;
	double v, g;
	std::vector<double> largest_up(activeEx);
	std::vector<double> largest_down(activeEx);
	double l_up = -1e100;
	double l_down = 1e100;
	for (e=0; e<(int)activeEx; e++)
	{
		largest_up[e] = -1e100;
		largest_down[e] = 1e100;
	}

	for (a = 0; a < (int)activeVar; a++)
	{
		e = variable[a].example;
		double v = alpha(a);
		double g = gradient(a);

		if (v < boxMax(a))
		{
			if (g > largest_up[e])
			{
				largest_up[e] = g;
				if (g > l_up) l_up = g;
			}
		}
		if (v > boxMin(a))
		{
			if (g < largest_down[e])
			{
				largest_down[e] = g;
				if (g < l_down) l_down = g;
			}
		}
	}

	if (! bUnshrinked)
	{
		if (l_up - l_down < 10.0 * epsilon)
		{
			// unshrink the problem at this accuracy level
			Unshrink(false);
			bUnshrinked = true;
			return;
		}
	}

	// shrink variables
	bool se = false;
	for (a = activeVar - 1; a >= 0; a--)
	{
		v = alpha(a);
		g = gradient(a);
		e = variable[a].example;

		if ((v == boxMin(a) && g <= largest_down[e]) || (v == boxMax(a) && g >= largest_up[e]))
		{
			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			DeactivateVariable(a);
			if (example[e].active == 0) se = true;
		}
	}

	if (se)
	{
		// exchange examples such that shrinked examples
		// are moved to the ends of the lists
		for (e = activeEx - 1; e >= 0; e--)
		{
			if (example[e].active == 0) DeactivateExample(e);
		}

		// shrink the corresponding cache entries
		for (e = 0; e < (int)activeEx; e++)
		{
			if (kernelMatrix.getCacheRowSize(e) > activeEx) kernelMatrix.CacheRowResize(e, activeEx);
		}
	}
}

void QpMcStzDecomp::Unshrink(bool complete)
{
	if (activeVar == variables) return;

	unsigned int i, a;
	float* q;
	double v;

	// compute the inactive gradient components (quadratic time complexity)
	for (a = activeVar; a < variables; a++) gradient(a) = linear(a);
	for (i = 0; i < variables; i++)
	{
		v = alpha(i);
		if (v == 0.0) continue;

		unsigned int ei = variable[i].example;
		unsigned int vi = variable[i].label;
		unsigned int yi = example[ei].label;
		q = kernelMatrix.Row(ei, 0, examples, true);
		for (a = activeVar; a < variables; a++)
		{
			unsigned int ea = variable[a].example;
			unsigned int va = variable[a].label;
			unsigned int ya = example[ea].label;

			double km = Modifier(yi, ya, vi, va);
			double k = q[ea];
			gradient(a) -= km * k * v;
		}
	}

	for (i = 0; i < examples; i++) example[i].active = classes;
	activeEx = examples;
	activeVar = variables;

	if (! complete) Shrink();
}

void QpMcStzDecomp::DeactivateVariable(unsigned int v)
{
	unsigned int ev = variable[v].example;
	unsigned int iv = variable[v].index;
	tExample* exv = &example[ev];
	unsigned int ih = exv->active - 1;
	unsigned int h = exv->variables[ih];
	variable[v].index = ih;
	variable[h].index = iv;
	std::swap(exv->variables[iv], exv->variables[ih]);
	iv = ih;
	exv->active--;

	unsigned int j = activeVar - 1;
	unsigned int ej = variable[j].example;
	unsigned int ij = variable[j].index;
	tExample* exj = &example[ej];

	// exchange entries in the lists
	XCHG_A(double, boxMin, v, j);
	XCHG_A(double, boxMax, v, j);
	XCHG_A(double, linear, v, j);
	XCHG_A(double, alpha, v, j);
	XCHG_A(double, gradient, v, j);
	XCHG_V(tVariable, variable, v, j);

	std::swap(exv->variables[iv], exj->variables[ij]);

	activeVar--;
}

void QpMcStzDecomp::DeactivateExample(unsigned int e)
{
	unsigned int j = activeEx - 1;

	XCHG_V(tExample, example, e, j);

	unsigned int v;
	unsigned int* pe = example[e].variables;
	unsigned int* pj = example[j].variables;
	for (v = 0; v < classes; v++)
	{
		variable[pe[v]].example = e;
		variable[pj[v]].example = j;
	}

	// notify the matrix cache
	kernelMatrix.CacheRowRelease(e);
	kernelMatrix.FlipColumnsAndRows(e, j);

	activeEx--;
}
