//===========================================================================
/*!
 *  \file QpBoxDecomp.cpp
 *
 *  \brief Quadratic programming for binary Support Vector Machines with box constraints
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


////////////////////////////////////////////////////////////////////////////////


QpBoxDecomp::QpBoxDecomp(CachedMatrix& quadraticPart)
: quadratic(quadraticPart)
{
	dimension = quadratic.getMatrixSize();

	// prepare lists
	alpha.resize(dimension, false);
	diagonal.resize(dimension, false);
	permutation.resize(dimension, false);
	gradient.resize(dimension, false);
	linear.resize(dimension, false);
	boxMin.resize(dimension, false);
	boxMax.resize(dimension, false);

	// prepare the permutation and the diagonal
	unsigned int i;
	for (i = 0; i < dimension; i++)
	{
		permutation(i) = i;
		diagonal(i) = quadratic.Entry(i, i);
	}

	maxIter = -1;
	WSS_Strategy = 1;
}

QpBoxDecomp::~QpBoxDecomp()
{
}


void QpBoxDecomp::Solve(const Array<double>& linearPart,
							   const Array<double>& boxLower,
							   const Array<double>& boxUpper,
							   Array<double>& solutionVector,
							   double eps)
{
	SIZE_CHECK(linearPart.ndim() == 1);
	SIZE_CHECK(boxLower.ndim() == 1);
	SIZE_CHECK(boxUpper.ndim() == 1);
	SIZE_CHECK(linearPart.dim(0) == dimension);
	SIZE_CHECK(boxLower.dim(0) == dimension);
	SIZE_CHECK(boxUpper.dim(0) == dimension);

	unsigned int a, i, j;
	float* q;

	for (i = 0; i < dimension; i++)
	{
		j = permutation(i);
		alpha(i) = solutionVector(j);
		linear(i) = linearPart(j);
		boxMin(i) = boxLower(j);
		boxMax(i) = boxUpper(j);
	}

	epsilon = eps;

	// prepare the solver internal variables
	active = dimension;
	gradient = linear;

	for (i = 0; i < dimension; i++)
	{
		if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpBoxDecomp::Solve] The feasible region is empty.");

		double v = alpha(i);
		if (v != 0.0)
		{
			q = quadratic.Row(i, 0, dimension);
			for (a = 0; a < dimension; a++) gradient(a) -= q[a] * v;
		}
	}

	// initial shrinking, e.g., for dummy variables
	Shrink();

	// decomposition loop
	Loop();

	// return alpha
	for (i = 0; i < dimension; i++)
	{
		solutionVector(permutation(i)) = alpha(i);
	}
}

// Compute the optimal step mu given the current point
// alpha, the gradient g, and the quadratic term Q in the
// interval [L, U]. Return the corresponding gain.
double QpBoxDecomp::StepEdge(double alpha, double g, double Q, double L, double U, double& mu)
{
	// compute the optimal unconstrained step
	double muHat = g / Q;

	// check for numerical problems
	if (! finite(muHat))
	{
		if (g > 0.0) mu = U - alpha;
		else mu = L - alpha;
		return 1e100;
	}

	// compute the optimal constrained step
	if (muHat < L - alpha) mu = L - alpha;
	else if (muHat > U - alpha) mu = U - alpha;
	else mu = muHat;

	// compute (twice) the gain
	double deltaMu = muHat - mu;
	return (muHat * muHat - deltaMu * deltaMu) * Q;
}

// Compute the optimal step (mui, muj) given the current
// point (alphai, alphaj), the gradient (gi, gj), and the
// symmetric positive semi definite matrix (Qii, Qij; Qij, Qjj)
// in the square [Li, Ui] x [Lj, Uj].
void QpBoxDecomp::Solve2D(double alphai, double alphaj,					// point
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

void QpBoxDecomp::Loop()
{
	unsigned int a, i,j;
	float* q;

	bUnshrinked = false;
	unsigned int shrinkCounter = (active < 1000) ? active : 1000;

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
				shrinkCounter = (active < 1000) ? active : 1000;

				SelectWorkingSet(i);
			}

			// update
			{
				double ai = alpha(i);
				double Li = boxMin(i);
				double Ui = boxMax(i);

				// update alpha, that is, solve the sub-problem defined by i
				double numerator = gradient(i);
				double denominator = diagonal(i);
				double mu = numerator / denominator;
				if (ai + mu < Li) mu = Li - ai;
				else if (ai + mu > Ui) mu = Ui - ai;
				alpha(i) += mu;

				// get the matrix row corresponding to the working set
				q = quadratic.Row(i, 0, active);

				// update the gradient
				for (a = 0; a < active; a++) gradient(a) -= mu * q[a];
			}

			shrinkCounter--;
			if (shrinkCounter == 0)
			{
				// shrink the problem
				Shrink();

				shrinkCounter = (active < 1000) ? active : 1000;
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
				shrinkCounter = (active < 1000) ? active : 1000;

				SelectWorkingSet(i, j);
			}

			// update
			{
				double ai = alpha(i);
				double Li = boxMin(i);
				double Ui = boxMax(i);

				double aj = alpha(j);
				double Lj = boxMin(j);
				double Uj = boxMax(j);

				// get the matrix rows corresponding to the working set
				float* q_i = quadratic.Row(i, 0, active);
				float* q_j = quadratic.Row(j, 0, active);

				// get the Q-matrix restricted to the working set
				double Qii = diagonal(i);
				double Qjj = diagonal(j);
				double Qij = q_i[j];

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

				// update the gradient
				for (a = 0; a < active; a++) gradient(a) -= (mu_i * q_i[a] + mu_j * q_j[a]);
			}

			shrinkCounter--;
			if (shrinkCounter == 0)
			{
				// shrink the problem
				Shrink();

				shrinkCounter = (active < 1000) ? active : 1000;
			}

			iter++;
		}
	}

	if (iter == maxIter) optimal = false;
}

bool QpBoxDecomp::SelectWorkingSet(unsigned int& i)
{
	double largest = 0.0;
	unsigned int a;

	for (a = 0; a < active; a++)
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

bool QpBoxDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	double maxGrad = 0.0;
	unsigned int a;

	// select first variable i
	// with first order method
	for (a = 0; a < active; a++)
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

	double gi = gradient(i);
	float* q = quadratic.Row(i, 0, active);
	double Qii = diagonal(i);

	// select second variable j
	// with second order method
	double maxGain = 0.0;
	for (a=0; a<active; a++)
	{
		if (a == i) continue;

		double ga = gradient(a);
		if ((alpha(a) > boxMin(a) && ga < 0.0)
				|| (alpha(a) < boxMax(a) && ga > 0.0))
		{
			double ga = gradient(a);
			double Qia = q[a];
			double Qaa = diagonal(a);

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

void QpBoxDecomp::Shrink()
{
	std::vector<unsigned int> shrinked;
	unsigned int a;
	double v, g;

	if (! bUnshrinked)
	{
		double largest = 0.0;
		for (a = 0; a < active; a++)
		{
			if (alpha(a) < boxMax(a))
			{
				if (gradient(a) > largest)
				{
					largest = gradient(a);
				}
			}
			if (alpha(a) > boxMin(a))
			{
				if (-gradient(a) > largest)
				{
					largest = -gradient(a);
				}
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

	// identify the variables to shrink
	for (a = 0; a < active; a++)
	{
		v = alpha(a);
		g = gradient(a);

		if ((v == boxMin(a) && g <= 0.0) || (v == boxMax(a) && g >= 0.0))
		{
			// In this moment no feasible step including this variable
			// can improve the objective. Thus deactivate the variable.
			shrinked.push_back(a);
			if (quadratic.getCacheRowSize(a) > 0) quadratic.CacheRowRelease(a);
		}
	}

	int s, sc = shrinked.size();
	if (sc == 0) return;

	unsigned int new_active = active - sc;

	// exchange variables such that shrinked variables
	// are moved to the ends of the lists.
	unsigned int k, high = active;
	for (s = sc - 1; s >= 0; s--)
	{
		k = shrinked[s];
		high--;

		// exchange the variables "k" and "high"
		FlipCoordinates(k, high);
	}

	// shrink the cache entries
	for (a = 0; a < active; a++)
	{
		if (quadratic.getCacheRowSize(a) > new_active) quadratic.CacheRowResize(a, new_active);
	}

	active = new_active;
}

void QpBoxDecomp::Unshrink(bool complete)
{
	if (active == dimension) return;

	unsigned int i, a;
	float* q;
	double v;

	// compute the inactive gradient components (quadratic time complexity)
	for (a = active; a < dimension; a++) gradient(a) = linear(a);
	for (i = 0; i < dimension; i++)
	{
		v = alpha(i);
		if (v == 0.0) continue;

		q = quadratic.Row(i, active, dimension, true);
		for (a = active; a < dimension; a++) gradient(a) -= q[a] * v;
	}

	active = dimension;
	if (! complete) Shrink();
}

void QpBoxDecomp::FlipCoordinates(unsigned int i, unsigned int j)
{
	if (i == j) return;

	// exchange entries in the simple lists
	XCHG_A(double, boxMin, i, j);
	XCHG_A(double, boxMax, i, j);
	XCHG_A(double, linear, i, j);
	XCHG_A(double, alpha, i, j);
	XCHG_A(unsigned int, permutation, i, j);
	XCHG_A(double, diagonal, i, j);
	XCHG_A(double, gradient, i, j);

	// notify the matrix cache
	quadratic.FlipColumnsAndRows(i, j);
}
