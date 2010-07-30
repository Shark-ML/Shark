//===========================================================================
/*!
 *  \file QuadraticProgram.cpp
 *
 *  \brief Quadratic programming for Support Vector Machines
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par Copyright (c) 1999-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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
#include <time.h>
#include <iostream>
#include <iomanip>


using namespace std;


// useful exchange macros for Array<T> and std::vector<T>
#define XCHG_A(t, a, i, j) {t temp; temp = a(i); a(i) = a(j); a(j) = temp;}
#define XCHG_V(t, a, i, j) {t temp; temp = a[i]; a[i] = a[j]; a[j] = temp;}



////////////////////////////////////////////////////////////////////////////////


QPSolver::QPSolver()
{
}

QPSolver::~QPSolver()
{
}


////////////////////////////////////////////////////////////////////////////////


QPMatrix::QPMatrix(unsigned int size)
{
	matrixsize = size;
}

QPMatrix::~QPMatrix()
{
}


////////////////////////////////////////////////////////////////////////////////


KernelMatrix::KernelMatrix(KernelFunction* kernelfunction,
						   const Array<double>& data)
: QPMatrix(data.dim(0))
, kernel(kernelfunction)
{
	x.resize(matrixsize, false);
	unsigned int i;
	for (i = 0; i < matrixsize; i++)
	{
		x(i) = new ArrayReference<double>(data[i]);
	}
}

KernelMatrix::~KernelMatrix()
{
	unsigned int i;
	for (i = 0; i < matrixsize; i++)
	{
		delete x(i);
	}
}


float KernelMatrix::Entry(unsigned int i, unsigned int j)
{
	return (float)kernel->eval(*x(i), *x(j));
}

void KernelMatrix::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	XCHG_A(ArrayReference<double>*, x, i, j);
}


////////////////////////////////////////////////////////////////////////////////


PrecomputedKernelMatrix::PrecomputedKernelMatrix(KernelFunction* kernelfunction,
				const Array<double>& data)
: KernelMatrix(kernelfunction, data)
{
	unsigned int i, j, N = data.dim(0);
	permutation.resize(N);
	matrix.resize(N, N, false);
	for (i=0; i<N; i++)
	{
		permutation[i] = i;
		for (j=0; j<=i; j++)
		{
			float k = kernelfunction->eval(data[i], data[j]);
			matrix(i, j) = matrix(j, i) = k;
		}
	}
}

PrecomputedKernelMatrix::~PrecomputedKernelMatrix()
{
}


float PrecomputedKernelMatrix::Entry(unsigned int i, unsigned int j)
{
	return matrix(permutation[i], permutation[j]);
}

void PrecomputedKernelMatrix::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	std::swap(permutation[i], permutation[j]);
}


////////////////////////////////////////////////////////////////////////////////


RegularizedKernelMatrix::RegularizedKernelMatrix(KernelFunction* kernel,
		const Array<double>& data,
		const Array<double>& diagModification)
		: KernelMatrix(kernel, data)
{
	SIZE_CHECK(data.dim(0) == diagModification.dim(0));
	diagMod = diagModification;
}

RegularizedKernelMatrix::~RegularizedKernelMatrix()
{
}


float RegularizedKernelMatrix::Entry(unsigned int i, unsigned int j)
{
	float ret = (float)KernelMatrix::Entry(i, j);
	if (i == j) ret += (float)diagMod(i);
	return ret;
}

void RegularizedKernelMatrix::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	KernelMatrix::FlipColumnsAndRows(i, j);

	XCHG_A(double, diagMod, i, j);
}


////////////////////////////////////////////////////////////////////////////////


QPMatrix2::QPMatrix2(QPMatrix* base)
: QPMatrix(2 * base->getMatrixSize())
{
	baseMatrix = base;

	mapping.resize(matrixsize, false);

	unsigned int i, ic = base->getMatrixSize();
	for (i = 0; i < ic; i++)
	{
		mapping(i) = i;
		mapping(i + ic) = i;
	}
}

QPMatrix2::~QPMatrix2()
{
	delete baseMatrix;
}


float QPMatrix2::Entry(unsigned int i, unsigned int j)
{
	return baseMatrix->Entry(mapping(i), mapping(j));
}

void QPMatrix2::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	XCHG_A(unsigned int, mapping, i, j);
}


////////////////////////////////////////////////////////////////////////////////


CachedMatrix::CachedMatrix(QPMatrix* base, unsigned int cachesize)
		: QPMatrix(base->getMatrixSize())
{
	baseMatrix = base;

	cacheSize = 0;
	cacheMaxSize = cachesize;
	cacheNewest = -1;
	cacheOldest = -1;

	cacheEntry.resize(matrixsize);
	unsigned int i;
	for (i = 0; i < matrixsize; i++)
	{
		cacheEntry[i].data = NULL;
		cacheEntry[i].length = 0;
		cacheEntry[i].older = -2;
		cacheEntry[i].newer = -2;
	}
	if (cachesize < 2 * matrixsize) throw SHARKEXCEPTION("[CachedMatrix::CachedMatrix] invalid cache size");

}

CachedMatrix::~CachedMatrix()
{
	cacheClear();
}


float* CachedMatrix::Row(unsigned int k, unsigned int begin, unsigned int end, bool temp)
{
	if (temp)
	{
		// return temporary data
		cacheTemp.resize(end);
		if (cacheEntry[k].length > (int)begin)
		{
			memcpy(&cacheTemp[0] + begin, cacheEntry[k].data + begin, sizeof(float) * (cacheEntry[k].length - begin));
			begin = cacheEntry[k].length;
		}
		unsigned int col;
		for (col = begin; col < end; col++) cacheTemp[col] = baseMatrix->Entry(k, col);
		return &cacheTemp[0];
	}
	else
	{
		// the data will be stored in the cache
		unsigned int l = cacheEntry[k].length;
		while (cacheSize + end > cacheMaxSize + l)
		{
			if (cacheOldest == (int)k)
			{
				cacheRemove(k);
				cacheAppend(k);
			}
			cacheDelete(cacheOldest);
		}
		if (l == 0)
		{
			cacheAdd(k, end);
		}
		else
		{
			cacheResize(k, end);
			if ((int)k != cacheNewest)
			{
				cacheRemove(k);
				cacheAppend(k);
			}
		}

		// compute remaining entries
		if (l < end)
		{
			float* p = cacheEntry[k].data + l;
			unsigned int col;
			for (col = l; col < end; col++)
			{
				*p = baseMatrix->Entry(k, col);
				p++;
			}
		}

		return cacheEntry[k].data;
	}
}

float CachedMatrix::Entry(unsigned int i, unsigned int j)
{
	return baseMatrix->Entry(i, j);
}

void CachedMatrix::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	int t;

	baseMatrix->FlipColumnsAndRows(i, j);

	// update the ordered cache list predecessors and successors
	t = cacheEntry[i].older;
	if (t != -2)
	{
		if (t == -1) cacheOldest = j;
		else cacheEntry[t].newer = j;
		t = cacheEntry[i].newer;
		if (t == -1) cacheNewest = j;
		else cacheEntry[t].older = j;
	}
	t = cacheEntry[j].older;
	if (cacheEntry[j].older != -2)
	{
		if (t == -1) cacheOldest = i;
		else cacheEntry[t].newer = i;
		t = cacheEntry[j].newer;
		if (t == -1) cacheNewest = i;
		else cacheEntry[t].older = i;
	}

	// exchange the cache entries
	XCHG_V(tCacheEntry, cacheEntry, i, j);

	// exchange all cache row entries
	unsigned int k, l;
	for (k = 0; k < matrixsize; k++)
	{
		l = cacheEntry[k].length;
		if (j < l)
		{
			XCHG_V(float, cacheEntry[k].data, i, j);
		}
		else if (i < l)
		{
			// only one element is available from the cache
			cacheEntry[k].data[i] = baseMatrix->Entry(k, i);
		}
	}
}

void CachedMatrix::CacheRowResize(unsigned int k, unsigned int newsize)
{
	if (cacheEntry[k].data == NULL) cacheAdd(k, newsize);
	else cacheResize(k, newsize);
}

void CachedMatrix::CacheRowRelease(unsigned int k)
{
	if (cacheEntry[k].data != NULL) cacheDelete(k);
}

void CachedMatrix::cacheAppend(int var)
{
	if (cacheNewest == -1)
	{
		cacheNewest = var;
		cacheOldest = var;
		cacheEntry[var].older = -1;
		cacheEntry[var].newer = -1;
	}
	else
	{
		cacheEntry[cacheNewest].newer = var;
		cacheEntry[var].older = cacheNewest;
		cacheEntry[var].newer = -1;
		cacheNewest = var;
	}
}

void CachedMatrix::cacheRemove(int var)
{
	if (cacheEntry[var].older == -1)
		cacheOldest = cacheEntry[var].newer;
	else
		cacheEntry[cacheEntry[var].older].newer = cacheEntry[var].newer;

	if (cacheEntry[var].newer == -1)
		cacheNewest = cacheEntry[var].older;
	else
		cacheEntry[cacheEntry[var].newer].older = cacheEntry[var].older;

	cacheEntry[var].older = -2;
	cacheEntry[var].newer = -2;
}

void CachedMatrix::cacheAdd(int var, unsigned int length)
{
	cacheEntry[var].length = length;
	cacheEntry[var].data = (float*)(void*)malloc(length * sizeof(float));
	if (cacheEntry[var].data == NULL) throw SHARKEXCEPTION("[CachedMatrix::cacheAppend] out of memory error");
	cacheSize += length;

	cacheAppend(var);
}

void CachedMatrix::cacheDelete(int var)
{
	free(cacheEntry[var].data);
	cacheSize -= cacheEntry[var].length;

	cacheEntry[var].data = NULL;
	cacheEntry[var].length = 0;

	cacheRemove(var);
}

void CachedMatrix::cacheResize(int var, unsigned int newlength)
{
	int diff = newlength - cacheEntry[var].length;
	if (diff == 0) return;
	cacheSize += diff;
	cacheEntry[var].length = newlength;
	cacheEntry[var].data = (float*)(void*)realloc((void*)cacheEntry[var].data, newlength * sizeof(float));
	if (cacheEntry[var].data == NULL) throw SHARKEXCEPTION("[CachedMatrix::cacheResize] out of memory error");
}

void CachedMatrix::cacheClear()
{
	unsigned int e, ec = cacheEntry.size();
	for (e = 0; e < ec; e++)
	{
		if (cacheEntry[e].data != NULL) free(cacheEntry[e].data);
		cacheEntry[e].data = NULL;
		cacheEntry[e].length = 0;
		cacheEntry[e].older = -2;
		cacheEntry[e].newer = -2;
	}
	cacheOldest = -1;
	cacheNewest = -1;
	cacheSize = 0;
}


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
			double Li = boxMin(i);
			double Ui = boxMax(i);
			double Lj = boxMin(j);
			double Uj = boxMax(j);

			// get the matrix rows corresponding to the working set
			qi = quadratic.Row(i, 0, active);
			qj = quadratic.Row(j, 0, active);

			// update alpha, that is, solve the sub-problem defined by i and j
			double numerator = gradient(i) - gradient(j);
			double denominator = diagonal(i) + diagonal(j) - 2.0 * qi[j];
			double mu = numerator / denominator;
			if (ai + mu < Li) mu = Li - ai;
			else if (ai + mu > Ui) mu = Ui - ai;
			if (aj - mu < Lj) mu = aj - Lj;
			else if (aj - mu > Uj) mu = aj - Uj;
			alpha(i) += mu;
			alpha(j) -= mu;

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

	// MVP stopping condition
	return (largestUp - smallestDown < epsilon);
}

bool QpSvmDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	// dynamic working set selection call
	bool ret = (this->*(this->currentWSS))(i, j);

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
				if (SelectWorkingSet(i,j))
				{
					optimal = true;
					break;
				}

				// shrink again
				Shrink();
				shrinkCounter = (activeVar < 1000) ? activeVar : 1000;
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
