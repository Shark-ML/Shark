//===========================================================================
/*!
 *  \file QpSvmDecomp.cpp
 *
 *  \brief Quadratic programming for binary Support Vector Machines
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

#define NOMINMAX

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


QpSvmDecomp::QpSvmDecomp(CachedMatrix& quadraticPart)
: quadratic(quadraticPart)
{
	printInfo = false;
	WSS_Strategy = NULL;
	useShrinking = true;
	maxIter = -1;
	maxSeconds = -1;

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
}

QpSvmDecomp::~QpSvmDecomp()
{
}


double QpSvmDecomp::Solve(const Array<double>& linearPart,
							   const Array<double>& boxLower,
							   const Array<double>& boxUpper,
							   Array<double>& solutionVector,
							   double eps,
							   double threshold)
{
	SIZE_CHECK(linearPart.ndim() == 1);
	SIZE_CHECK(boxLower.ndim() == 1);
	SIZE_CHECK(boxUpper.ndim() == 1);
	SIZE_CHECK(linearPart.dim(0) == dimension);
	SIZE_CHECK(boxLower.dim(0) == dimension);
	SIZE_CHECK(boxUpper.dim(0) == dimension);

	unsigned int a, i, j;
	float* qi;
	float* qj;

	for (i = 0; i < dimension; i++)
	{
		j = permutation(i);
		alpha(i) = solutionVector(j);
		linear(i) = linearPart(j);
		boxMin(i) = boxLower(j);
		boxMax(i) = boxUpper(j);
	}

	epsilon = eps;
	optimal = false;

	// prepare the solver internal variables
	active = dimension;
	gradient = linear;

	for (i = 0; i < dimension; i++)
	{
		if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpSvmDecomp::Solve] The feasible region is empty.");

		double v = alpha(i);
		if (v != 0.0)
		{
			qi = quadratic.Row(i, 0, dimension);
			for (a = 0; a < dimension; a++) gradient(a) -= qi[a] * v;
		}
	}

	bFirst = true;
	bUnshrinked = false;
	unsigned int shrinkCounter = (active < 1000) ? active : 1000;

	SelectWSS();

	// decomposition loop
	if (printInfo) cout << "{" << flush;
	iter = 0;
	time_t starttime;
	time(&starttime);
	while (iter != maxIter)
	{
		// select a working set and check for optimality
		if (SelectWorkingSet(i, j))
		{
			// seems to be optimal
			if (printInfo) cout << "*" << flush;

			if (! useShrinking)
			{
				optimal = true;
				break;
			}

			// do costly unshrinking
			Unshrink();
			shrinkCounter = 1;

			// check again on the whole problem
			if (SelectWorkingSet(i, j))
			{
				optimal = true;
				break;
			}
		}

		// SMO update
		{
			double ai = alpha(i);
			double aj = alpha(j);
			double Ui = boxMax(i);
			double Lj = boxMin(j);

			// get the matrix rows corresponding to the working set
			qi = quadratic.Row(i, 0, active);
			qj = quadratic.Row(j, 0, active);

			// update alpha, that is, solve the sub-problem defined by i and j
			double numerator = gradient(i) - gradient(j);
			double denominator = diagonal(i) + diagonal(j) - 2.0 * qi[j];
			double mu = numerator / denominator;
			
			// do the update carefully - avoid numerical problems
			if (mu >= std::min(Ui - ai, aj - Lj))
			{
				if (Ui - ai > aj - Lj)
				{
					mu = aj - Lj;
					alpha(i) += mu;
					alpha(j) = Lj;
				}
				else if (Ui - ai < aj - Lj)
				{
					mu = Ui - ai;
					alpha(i) = Ui;
					alpha(j) -= mu;
				}
				else
				{
					mu = Ui - ai;
					alpha(i) = Ui;
					alpha(j) = Lj;
				}
			}
			else
			{
				alpha(i) += mu;
				alpha(j) -= mu;
			}

			// update the gradient
			for (a = 0; a < active; a++) gradient(a) -= mu * (qi[a] - qj[a]);
		}

		shrinkCounter--;
		if (shrinkCounter == 0)
		{
			// shrink the problem
			if (useShrinking) Shrink();

			shrinkCounter = (active < 1000) ? active : 1000;

			if (maxSeconds != -1)
			{
				time_t currenttime;
				time(&currenttime);
				if (currenttime - starttime > maxSeconds) break;
			}
		}

		if (threshold < 1e100)
		{
			double objective = 0.0;
			for (i = 0; i < active; i++)
			{
				objective += (gradient(i) + linear(i)) * alpha(i);
			}
			objective *= 0.5;
			if (objective > threshold) break;
		}

		iter++;
		if (printInfo)
		{
			if ((iter & 1023) == 0) cout << "." << flush;
		}
	}
	if (printInfo) cout << endl << "} #iterations=" << (long int)iter << endl;

	Unshrink(true);

	// return alpha and the objective value
	double objective = 0.0;
	for (i = 0; i < dimension; i++)
	{
		solutionVector(permutation(i)) = alpha(i);
		objective += (gradient(i) + linear(i)) * alpha(i);
	}
	return 0.5 * objective;
}

double QpSvmDecomp::ComputeInnerProduct(unsigned int index, const Array<double>& coeff)
{
	unsigned int e = permutation(index);
	unsigned int i;
	unsigned int j;
	double ret = 0.0;
	double c;
	unsigned int crs_j, crs_e = quadratic.getCacheRowSize(e);

	for (i = 0; i < dimension; i++)
	{
		j = permutation(i);
		c = coeff(j);
		if (c != 0.0)
		{
			if (j < crs_e) ret += c * quadratic.Row(e, 0, crs_e)[j];
			else
			{
				crs_j = quadratic.getCacheRowSize(j);
				if (e < crs_j) ret += c * quadratic.Row(j, 0, crs_j)[e];
				else ret += c * quadratic.Entry(j, e);
			}
		}
	}

	return ret;
}

void QpSvmDecomp::getGradient(Array<double>& grad)
{
	grad.resize(dimension, false);
	unsigned int i;
	for (i = 0; i < dimension; i++) grad(permutation(i)) = gradient(i);
}

bool QpSvmDecomp::MVP(unsigned int& i, unsigned int& j)
{
	double largestUp = -1e100;
	double smallestDown = 1e100;
	unsigned int a;

	for (a = 0; a < active; a++)
	{
		if (alpha(a) < boxMax(a))
		{
			if (gradient(a) > largestUp)
			{
				largestUp = gradient(a);
				i = a;
			}
		}
		if (alpha(a) > boxMin(a))
		{
			if (gradient(a) < smallestDown)
			{
				smallestDown = gradient(a);
				j = a;
			}
		}
	}

	// MVP stopping condition
	return (largestUp - smallestDown < epsilon);
}

bool QpSvmDecomp::HMG(unsigned int& i, unsigned int& j)
{
	if (bFirst)
	{
		// the cache is empty - use MVP
		bFirst = false;
// 		return MVP(i, j);			// original paper: use MVP
		return Libsvm28(i, j);		// better: use second order algorithm
	}

	// check the corner condition
	{
		double Li = boxMin(old_i);
		double Ui = boxMax(old_i);
		double Lj = boxMin(old_j);
		double Uj = boxMax(old_j);
		double eps_i = 1e-8 * (Ui - Li);
		double eps_j = 1e-8 * (Uj - Lj);
		if ((alpha(old_i) <= Li + eps_i || alpha(old_i) >= Ui - eps_i)
				&& ((alpha(old_j) <= Lj + eps_j || alpha(old_j) >= Uj - eps_j)))
		{
			if (printInfo) cout << "^" << flush;
//	 		return MVP(i, j);			// original paper: use MVP
			return Libsvm28(i, j);		// better: use second order algorithm
		}
	}

	// generic situation: use the MG selection
	unsigned int a;
	double aa, ab;					// alpha values
	double da, db;					// diagonal entries of Q
	double ga, gb;					// gradient in coordinates a and b
	double gain;
	double La, Ua, Lb, Ub;
	double denominator;
	float* q;
	double mu_max, mu_star;

	double best = 0.0;
	double mu_best = 0.0;

	// try combinations with b = old_i
	q = quadratic.Row(old_i, 0, active);
	ab = alpha(old_i);
	db = diagonal(old_i);
	Lb = boxMin(old_i);
	Ub = boxMax(old_i);
	gb = gradient(old_i);
	for (a = 0; a < active; a++)
	{
		if (a == old_i || a == old_j) continue;

		aa = alpha(a);
		da = diagonal(a);
		La = boxMin(a);
		Ua = boxMax(a);
		ga = gradient(a);

		denominator = (da + db - 2.0 * q[a]);
		mu_max = (ga - gb) / denominator;
		mu_star = mu_max;

		if (aa + mu_star < La) mu_star = La - aa;
		else if (mu_star + aa > Ua) mu_star = Ua - aa;
		if (ab - mu_star < Lb) mu_star = ab - Lb;
		else if (ab - mu_star > Ub) mu_star = ab - Ub;

		gain = mu_star * (2.0 * mu_max - mu_star) * denominator;

		// select the largest gain
		if (gain > best)
		{
			best = gain;
			mu_best = mu_star;
			i = a;
			j = old_i;
		}
	}

	// try combinations with old_j
	q = quadratic.Row(old_j, 0, active);
	ab = alpha(old_j);
	db = diagonal(old_j);
	Lb = boxMin(old_j);
	Ub = boxMax(old_j);
	gb = gradient(old_j);
	for (a = 0; a < active; a++)
	{
		if (a == old_i || a == old_j) continue;

		aa = alpha(a);
		da = diagonal(a);
		La = boxMin(a);
		Ua = boxMax(a);
		ga = gradient(a);

		denominator = (da + db - 2.0 * q[a]);
		mu_max = (ga - gb) / denominator;
		mu_star = mu_max;

		if (aa + mu_star < La) mu_star = La - aa;
		else if (mu_star + aa > Ua) mu_star = Ua - aa;
		if (ab - mu_star < Lb) mu_star = ab - Lb;
		else if (ab - mu_star > Ub) mu_star = ab - Ub;

		gain = mu_star * (2.0 * mu_max - mu_star) * denominator;

		// select the largest gain
		if (gain > best)
		{
			best = gain;
			mu_best = mu_star;
			i = a;
			j = old_j;
		}
	}

	// stopping condition
	return (fabs(mu_best) < epsilon);
}

bool QpSvmDecomp::Libsvm28(unsigned int& i, unsigned int& j)
{
	i = 0;
	j = 1;

	double largestUp = -1e100;
	double smallestDown = 1e100;
	unsigned int a;

	// find the first index of the MVP
	for (a = 0; a < active; a++)
	{
		if (alpha(a) < boxMax(a))
		{
			if (gradient(a) > largestUp)
			{
				largestUp = gradient(a);
				i = a;
			}
		}
	}

	// find the second index using second order information
	float* q = quadratic.Row(i, 0, active);
	double best = 0.0;
	for (a = 0; a < active; a++)
	{
		if (alpha(a) > boxMin(a))
		{
			if (gradient(a) < smallestDown) smallestDown = gradient(a);

			double grad_diff = largestUp - gradient(a);
			if (grad_diff > 0.0)
			{
				double quad_coef = diagonal(i) + diagonal(a) - 2.0 * q[a];
				if (quad_coef == 0.0) continue;
				double obj_diff = (grad_diff * grad_diff) / quad_coef;

				if (obj_diff > best)
				{
					best = obj_diff;
					j = a;
				}
			}
		}
	}
	
	if (best == 0.0) return true;		// numerical accuracy reached :(

	// MVP stopping condition
	return (largestUp - smallestDown < epsilon);
}

bool QpSvmDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	// dynamic working set selection call
	bool ret = (this->*(this->currentWSS))(i, j);
	if (gradient(i) < gradient(j)) std::swap(i, j);

	old_i = i;
	old_j = j;
	return ret;
}

void QpSvmDecomp::SelectWSS()
{
	if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "MVP") == 0)
	{
		// most violating pair, used e.g. in LIBSVM 2.71
		currentWSS = &QpSvmDecomp::MVP;
	}
	else if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "HMG") == 0)
	{
		// hybrid maximum gain, suitable for large problems
		currentWSS = &QpSvmDecomp::HMG;
	}
	else if (WSS_Strategy != NULL && strcmp(WSS_Strategy, "LIBSVM28") == 0)
	{
		// LIBSVM 2.8 second order algorithm
		currentWSS = &QpSvmDecomp::Libsvm28;
	}
	else
	{
		// default strategy:
		// use HMG as long as the problem does not fit into the cache,
		// use the LIBSVM 2.8 algorithm afterwards
		if (active * active > quadratic.getMaxCacheSize())
			currentWSS = &QpSvmDecomp::HMG;
		else
			currentWSS = &QpSvmDecomp::Libsvm28;
	}
}

void QpSvmDecomp::Shrink()
{
	double largestUp = -1e100;
	double smallestDown = 1e100;
	std::vector<unsigned int> shrinked;
	unsigned int a;
	double v, g;

	for (a = 0; a < active; a++)
	{
		v = alpha(a);
		g = gradient(a);
		if (v > boxMin(a))
		{
			if (g < smallestDown) smallestDown = g;
		}
		if (v < boxMax(a))
		{
			if (g > largestUp) largestUp = g;
		}
	}

	if (! bUnshrinked && (largestUp - smallestDown < 10.0 * epsilon))
	{
		// unshrink the problem at this accuracy level
		if (printInfo) cout << "#" << flush;
		Unshrink();
		bUnshrinked = true;
		SelectWSS();
		return;
	}

	// identify the variables to shrink
	for (a = 0; a < active; a++)
	{
		if (a == old_i) continue;
		if (a == old_j) continue;
		v = alpha(a);
		g = gradient(a);

		if (v == boxMin(a))
		{
			if (g > smallestDown) continue;
		}
		else if (v == boxMax(a))
		{
			if (g < largestUp) continue;
		}
		else continue;

		// In this moment no feasible step including this variable
		// can improve the objective. Thus deactivate the variable.
		shrinked.push_back(a);
		if (quadratic.getCacheRowSize(a) > 0) quadratic.CacheRowRelease(a);
	}

	int s, sc = shrinked.size();
	if (sc == 0)
	{
		return;
	}
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

	SelectWSS();
}

void QpSvmDecomp::Unshrink(bool complete)
{
	if (printInfo) cout << "[" << flush;
	if (active == dimension)
	{
		if (printInfo) cout << "]" << flush;
		return;
	}

	unsigned int i, a;
	float* q;
	double v, g;
	double largestUp = -1e100;
	double smallestDown = 1e100;

	// compute the inactive gradient components (quadratic time complexity)
	for (a = active; a < dimension; a++) gradient(a) = linear(a);
	for (i = 0; i < dimension; i++)
	{
		v = alpha(i);
		if (v == 0.0) continue;

		q = quadratic.Row(i, active, dimension, true);
		for (a = active; a < dimension; a++) gradient(a) -= q[a] * v;
	}

	if (complete)
	{
		active = dimension;
		if (printInfo) cout << "]" << flush;
		return;
	}

	// find largest KKT violations
	for (a = 0; a < dimension; a++)
	{
		g = gradient(a);
		v = alpha(a);

		if (v > boxMin(a) && g < smallestDown) smallestDown = g;
		if (v < boxMax(a) && g > largestUp) largestUp = g;
	}

	// identify the variables to activate
	for (a = active; a < dimension; a++)
	{
		if (a == old_i) continue;
		if (a == old_j) continue;
		g = gradient(a);
		v = alpha(a);

		if (v == boxMin(a))
		{
			if (g <= smallestDown) continue;
		}
		else if (v == boxMax(a))
		{
			if (g >= largestUp) continue;
		}

		FlipCoordinates(active, a);
		active++;
	}

	if (printInfo) cout << active << "]" << flush;
}

void QpSvmDecomp::FlipCoordinates(unsigned int i, unsigned int j)
{
	if (i == j) return;

	// check the previous working set
	if (old_i == i) old_i = j;
	else if (old_i == j) old_i = i;

	if (old_j == i) old_j = j;
	else if (old_j == j) old_j = i;

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
