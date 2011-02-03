//===========================================================================
/*!
 *  \file QpMcDecomp.cpp
 *
 *  \brief Quadratic programming for Multi-Class Support Vector Machines
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


QpMcDecomp::QpMcDecomp(CachedMatrix& kernel)
: kernelMatrix(kernel)
{
	examples = kernelMatrix.getMatrixSize();
	maxIter = -1;
	WSS_Strategy = 2;
}

QpMcDecomp::~QpMcDecomp()
{
}


// Compute the optimal step mu given the current point
// alpha, the gradient g, and the quadratic term Q in the
// interval [L, U]. Return the corresponding gain.
double QpMcDecomp::StepEdge(double alpha, double g, double Q, double L, double U, double& mu)
{
	// compute the optimal unconstrained step
	double muHat = g / Q;

	// compute the optimal constrained step
	if (muHat < L - alpha) mu = L - alpha;
	else if (muHat > U - alpha) mu = U - alpha;
	else mu = muHat;

	// compute (twice) the gain
	if (! finite(muHat)) return 1e100;
	double deltaMu = muHat - mu;
	return (muHat * muHat - deltaMu * deltaMu) * Q;
}

// Compute the optimal step (mui, muj) given the current
// point (alphai, alphaj), the gradient (gi, gj), and the
// symmetric positive semi definite matrix (Qii, Qij; Qij, Qjj)
// in the square [Li, Ui] x [Lj, Uj].
void QpMcDecomp::Solve2D(double alphai, double alphaj,					// point
							double gi, double gj,						// gradient
							double Qii, double Qij, double Qjj,			// Q-matrix
							double Li, double Ui, double Lj, double Uj,	// bounds
							double& mui, double& muj)					// step
{
	double QD = Qii * Qjj;
	double detQ = QD - Qij * Qij;
	if (detQ < 1e-10 * QD)
	{
		if (Qii == 0.0 && Qjj == 0.0)
		{
			// Q has rank zero (is the zero matrix)
			// just follow the gradient
			if (gi > 0.0) mui = Ui - alphai;
			else if (gi < 0.0) mui = Li - alphai;
			else mui = 0.0;
			if (gj > 0.0) muj = Uj - alphaj;
			else if (gj < 0.0) muj = Lj - alphaj;
			else muj = 0.0;
		}
		else
		{
			// Q has rank one
			double gamma = Qii * gj - Qij * gi;
			double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
			double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

			// edge with fixed mu_i
			if (Qij * gamma > 0.0)
			{
				edgei_mui = Li - alphai;
				edgei_gain = StepEdge(alphaj, gj - Qij * edgei_mui, Qjj, Lj, Uj, edgei_muj);
			}
			else if (Qij * gamma < 0.0)
			{
				edgei_mui = Ui - alphai;
				edgei_gain = StepEdge(alphaj, gj - Qij * edgei_mui, Qjj, Lj, Uj, edgei_muj);
			}

			// edge with fixed mu_j
			if (Qii * gamma < 0.0)
			{
				edgej_muj = Lj - alphaj;
				edgej_gain = StepEdge(alphai, gi - Qij * edgej_muj, Qii, Li, Ui, edgej_mui);
			}
			else if (Qii * gamma > 0.0)
			{
				edgej_muj = Uj - alphaj;
				edgej_gain = StepEdge(alphai, gi - Qij * edgej_muj, Qii, Li, Ui, edgej_mui);
			}

			// keep the better edge point
			if (edgei_gain > edgej_gain)
			{
				mui = edgei_mui;
				muj = edgei_muj;
			}
			else
			{
				mui = edgej_mui;
				muj = edgej_muj;
			}
		}
	}
	else
	{
		// Q has full rank of two, thus it is invertible
		double muiHat = (Qjj * gi - Qij * gj) / detQ;
		double mujHat = (Qii * gj - Qij * gi) / detQ;
		double edgei_mui = 0.0, edgei_muj = 0.0, edgei_gain = 0.0;
		double edgej_mui = 0.0, edgej_muj = 0.0, edgej_gain = 0.0;

		// edge with fixed mu_i
		if (muiHat < Li - alphai)
		{
			edgei_mui = Li - alphai;
			edgei_gain = StepEdge(alphaj, gj - Qij * edgei_mui, Qjj, Lj, Uj, edgei_muj);
		}
		else if (muiHat > Ui - alphai)
		{
			edgei_mui = Ui - alphai;
			edgei_gain = StepEdge(alphaj, gj - Qij * edgei_mui, Qjj, Lj, Uj, edgei_muj);
		}

		// edge with fixed mu_j
		if (mujHat < Lj - alphaj)
		{
			edgej_muj = Lj - alphaj;
			edgej_gain = StepEdge(alphai, gi - Qij * edgej_muj, Qii, Li, Ui, edgej_mui);
		}
		else if (mujHat > Uj - alphaj)
		{
			edgej_muj = Uj - alphaj;
			edgej_gain = StepEdge(alphai, gi - Qij * edgej_muj, Qii, Li, Ui, edgej_mui);
		}

		// keep the unconstrained optimum or the better edge point
		if (edgei_gain == 0.0 && edgej_gain == 0.0)
		{
			mui = muiHat;
			muj = mujHat;
		}
		else if (edgei_gain > edgej_gain)
		{
			mui = edgei_mui;
			muj = edgei_muj;
		}
		else
		{
			mui = edgej_mui;
			muj = edgej_muj;
		}
	}
}

void QpMcDecomp::Solve(unsigned int classes,
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
		if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpMcDecomp::Solve] The feasible region is empty.");

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
		if (WSS_Strategy == 1)
		{
			// select a working set and check for optimality
			if (SelectWorkingSet(i))
			{
				// seems to be optimal

				// do costly unshrinking
				Unshrink(true);

				// check again on the whole problem
				if (SelectWorkingSet(i))
				{
					optimal = true;
					break;
				}

				// shrink again
				Shrink();
				shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

				SelectWorkingSet(i);
			}

			// update
			{
				double ai = alpha(i);
				double Li = boxMin(i);
				double Ui = boxMax(i);
				unsigned int ei = variable[i].example;
				unsigned int vi = variable[i].label;
				unsigned int yi = example[ei].label;

				// update alpha, that is, solve the sub-problem defined by i
				double numerator = gradient(i);
				double denominator = variable[i].diagonal;
				double mu = numerator / denominator;

				if (ai + mu < Li) { mu = Li - ai; alpha(i) = Li; }
				else if (ai + mu > Ui) { mu = Ui - ai; alpha(i) = Ui; }
				else alpha(i) += mu;

				// get the matrix row corresponding to the working set
				q = kernelMatrix.Row(ei, 0, activeEx);

				// update the gradient
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
						gradient(j) -= mu * km * k;
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
		else if (WSS_Strategy == 2)
		{
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
				double Li = boxMin(i);
				double Ui = boxMax(i);
				unsigned int ei = variable[i].example;
				unsigned int vi = variable[i].label;
				unsigned int yi = example[ei].label;

				double aj = alpha(j);
				double Lj = boxMin(j);
				double Uj = boxMax(j);
				unsigned int ej = variable[j].example;
				unsigned int vj = variable[j].label;
				unsigned int yj = example[ej].label;

				// get the matrix rows corresponding to the working set
				float* q_i = NULL;
				float* q_j = NULL;
				q_i = kernelMatrix.Row(ei, 0, activeEx);
				q_j = kernelMatrix.Row(ej, 0, activeEx);

				// get the Q-matrix restricted to the working set
				double k = q_i[ej];
				double km = Modifier(yi, yj, vi, vj);
				double Qii = variable[i].diagonal;
				double Qjj = variable[j].diagonal;
				double Qij = km * k;

				// solve the sub-problem
				double mu_i = 0.0;
				double mu_j = 0.0;
				Solve2D(ai, aj,
						gradient(i), gradient(j),
						Qii, Qij, Qjj,
						Li, Ui, Lj, Uj,
						mu_i, mu_j);

				// update alpha
				alpha(i) += mu_i;
				alpha(j) += mu_j;

				// repair numerical inaccuracies
				if (alpha(i) - Li < 1e-12 * (Ui - Li)) alpha(i) = Li;
				else if (Ui - alpha(i) < 1e-12 * (Ui - Li)) alpha(i) = Ui;
				if (alpha(j) - Lj < 1e-12 * (Uj - Lj)) alpha(j) = Lj;
				else if (Uj - alpha(j) < 1e-12 * (Uj - Lj)) alpha(j) = Uj;

				// update the gradient
				for (a = 0; a < activeEx; a++)
				{
					double k_i = q_i[a];
					double k_j = q_j[a];
					tExample* ex = &example[a];
					unsigned int ya = ex->label;
					bc = ex->active;
					for (b = 0; b < bc; b++)
					{
						unsigned int p = ex->variables[b];
						unsigned int vp = variable[p].label;
						double km_i = Modifier(yi, ya, vi, vp);
						double km_j = Modifier(yj, ya, vj, vp);
						gradient(p) -= ((mu_i * km_i * k_i) + (mu_j * km_j * k_j));
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
	}

	if (iter == maxIter) optimal = false;

	// return alpha
	for (i = 0; i < variables; i++)
	{
		unsigned int j = classes * example[variable[i].example].index + variable[i].label;
		solutionVector(j) = alpha(i);
	}
}

bool QpMcDecomp::SelectWorkingSet(unsigned int& i)
{
	double largest = 0.0;
	unsigned int a;

	for (a = 0; a < activeVar; a++)
	{
		double v = alpha(a);
		double g = gradient(a);
		if (v < boxMax(a))
		{
			if (g > largest)
			{
				largest = g;
				i = a;
			}
		}
		if (v > boxMin(a))
		{
			if (-g > largest)
			{
				largest = -g;
				i = a;
			}
		}
	}

	// KKT stopping condition
	return (largest < epsilon);
}

bool QpMcDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	i = 0; j = 1;

	double maxGrad = 0.0;
	unsigned int a;

	// select first variable i
	// with first order method
	for (a = 0; a < activeVar; a++)
	{
		double v = alpha(a);
		double g = gradient(a);
		if (v < boxMax(a))
		{
			if (g > maxGrad)
			{
				maxGrad = g;
				i = a;
			}
		}
		if (v > boxMin(a))
		{
			if (-g > maxGrad)
			{
				maxGrad = -g;
				i = a;
			}
		}
	}

	// KKT stopping condition
	if (maxGrad < epsilon) return true;

	unsigned int ei = variable[i].example;
	unsigned int vi = variable[i].label;
	unsigned int yi = example[ei].label;
	double gi = gradient(i);

	float* q = kernelMatrix.Row(ei,0, activeEx);
	double Qii = variable[i].diagonal;

	// select second variable j
	// with second order method
	double maxGain = 0.0;
	for (a=0; a<activeVar; a++)
	{
		if (a == i) continue;

		double ga = gradient(a);
		if ((alpha(a) > boxMin(a) && ga < 0.0)
				|| (alpha(a) < boxMax(a) && ga > 0.0))
		{
			unsigned int ea = variable[a].example;
			unsigned int va = variable[a].label;
			unsigned int ya = example[ea].label;
			double ga = gradient(a);

			double k = q[ea];
			double km = Modifier(yi, ya, vi, va);
			double Qia = km * k;
			double Qaa = variable[a].diagonal;

			double QD = Qii * Qaa;
			double detQ = QD - Qia * Qia;
			if (detQ < 1e-10 * QD)
			{
				if (Qii == 0.0 && Qaa == 0.0)
				{
					// Q has rank zero
					if (gi != 0.0 || ga != 0.0)
					{
						j = a;
						return false;		// infinite gain, return immediately
					}
				}
				else
				{
					// Q has rank one
					if (Qii * ga - Qia * gi != 0.0)
					{
						j = a;
						return false;		// infinite gain, return immediately
					}
					else
					{
						double g2 = ga*ga + gi*gi;
						double gain = (g2*g2) / (ga*ga*Qaa + 2.0*ga*gi*Qia + gi*gi*Qii);
						if (gain > maxGain)
						{
							maxGain = gain;
							j = a;
						}
					}
				}
			}
			else
			{
				// Q has rank two
				double gain = (ga*ga*Qii - 2.0*ga*gi*Qia + gi*gi*Qaa) / detQ;
				if (gain > maxGain)
				{
					maxGain = gain;
					j = a;
				}
			}
		}
	}

	return false;		// solution is not optimal
}

void QpMcDecomp::Shrink()
{
	int a;
	double v, g;

	if (! bUnshrinked)
	{
		double largest = 0.0;
		for (a = 0; a < (int)activeVar; a++)
		{
			if (alpha(a) < boxMax(a))
			{
				if (gradient(a) > largest) largest = gradient(a);
			}
			if (alpha(a) > boxMin(a))
			{
				if (-gradient(a) > largest) largest = -gradient(a);
			}
		}
		if (largest < 10.0 * epsilon)
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

		if ((v == boxMin(a) && g <= 0.0) || (v == boxMax(a) && g >= 0.0))
		{
			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			unsigned int e = variable[a].example;
			DeactivateVariable(a);
			if (example[e].active == 0)
			{
				se = true;
			}
		}
	}

	if (se)
	{
		// exchange examples such that shrinked examples
		// are moved to the ends of the lists
		for (a = activeEx - 1; a >= 0; a--)
		{
			if (example[a].active == 0) DeactivateExample(a);
		}

		// shrink the corresponding cache entries
		for (a = 0; a < (int)activeEx; a++)
		{
			if (kernelMatrix.getCacheRowSize(a) > activeEx) kernelMatrix.CacheRowResize(a, activeEx);
		}
	}
}

void QpMcDecomp::Unshrink(bool complete)
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

void QpMcDecomp::DeactivateVariable(unsigned int v)
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

void QpMcDecomp::DeactivateExample(unsigned int e)
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
