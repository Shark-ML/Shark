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


InputLabelMatrix::InputLabelMatrix(KernelFunction* kernelfunction,
								   const Array<double>& input,
								   const Array<double>& target,
								   const Array<double>& prototypes)
		: QPMatrix(input.dim(0))
		, kernel(kernelfunction)
{
	unsigned int i, j, m, classes = prototypes.dim(0);

	x.resize(matrixsize, false);
	y.resize(matrixsize);
	for (i = 0; i < matrixsize; i++)
	{
		x(i) = new ArrayReference<double>(input[i]);
		y[i] = (unsigned int)target(i, 0);
	}

	prototypeProduct.resize(classes, classes, false);
	for (i = 0; i < classes; i++)
	{
		for (j = 0; j <= i; j++)
		{
			double scp = 0.0;
			for (m = 0; m < classes; m++) scp += prototypes(i, m) * prototypes(j, m);
			prototypeProduct(i, j) = prototypeProduct(j, i) = scp;
		}
	}
}

InputLabelMatrix::~InputLabelMatrix()
{
	unsigned int i;
	for (i = 0; i < matrixsize; i++) delete x(i);
}


float InputLabelMatrix::Entry(unsigned int i, unsigned int j)
{
	return (float)(kernel->eval(*x(i), *x(j)) * prototypeProduct(y[i], y[j]));
}

void InputLabelMatrix::FlipColumnsAndRows(unsigned int i, unsigned int j)
{
	XCHG_A(ArrayReference<double>*, x, i, j);
	XCHG_V(unsigned int, y, i, j);
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


void QpSvmCG::Solve(const Array<double>& quadratic,
					const Array<double>& linear,
					const Array<double>& boxMin,
					const Array<double>& boxMax,
					Array<double>& point,
					double accuracy)
{
	SIZE_CHECK(quadratic.ndim() == 2);
	SIZE_CHECK(linear.ndim() == 1);
	SIZE_CHECK(boxMin.ndim() == 1);
	SIZE_CHECK(boxMax.ndim() == 1);
	SIZE_CHECK(point.ndim() == 1);

	int dimension = quadratic.dim(0);

	SIZE_CHECK((int)quadratic.dim(1) == dimension);
	SIZE_CHECK((int)linear.dim(0) == dimension);
	SIZE_CHECK((int)boxMin.dim(0) == dimension);
	SIZE_CHECK((int)boxMax.dim(0) == dimension);
	SIZE_CHECK((int)point.dim(0) == dimension);

	int i, j;
	Array<double> direction(dimension);
	Array<bool> active(dimension);
	Array<double> gradient(dimension);
	Array<double> conjugate(dimension);
	conjugate = 0.0;
	double DirLen2 = 0.0;
	double DirLen2old = 0.0;
	double accuracy2 = accuracy * accuracy;
	bool changed, bounds_changed;
	int nActive = 0;
	int nUpdatesLeft = -1;
	int bound = -1;
	int nReset = 0;
	double step = 0.0;
	double clipped = 0.0;
	double value;
	double first = 0.0;
	double second = 0.0;
	double scp, sub;
	double norm2;

	active = false;

	int iter, maxiter = 10 * dimension * dimension;
	for (iter = 0; iter < maxiter; iter++)
	{
		if (nUpdatesLeft == -1)
		{
			// compute the gradient from scratch
			for (i = 0; i < dimension; i++)
			{
				value = linear(i);
				for (j = 0; j < dimension; j++) value += quadratic(i, j) * point(j);
				gradient(i) = value;
			}
			nUpdatesLeft = 100;
		}

		// compute the set of active inequality constraints
		scp = 0.0;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			scp += gradient(i);
		}
		sub = scp / (dimension - nActive);
		bounds_changed = false;
		do
		{
			changed = false;
			for (i = 0; i < dimension; i++)
			{
				if (active(i))
				{
					double g = gradient(i) - (scp + gradient(i)) / (dimension - nActive + 1.0);
					if ((g <= 0.0 && point(i) == boxMin(i))
							|| (g >= 0.0 && point(i) == boxMax(i)))
					{
						active(i) = false;
						nActive--;
						scp += gradient(i);
						sub = scp / (dimension - nActive);
						changed = true;
					}
				}
				else
				{
					double g = gradient(i) - sub;
					if ((g > 0.0 && point(i) == boxMin(i))
							|| (g < 0.0 && point(i) == boxMax(i)))
					{
						active(i) = true;
						nActive++;
						scp -= gradient(i);
						sub = scp / (dimension - nActive);
						changed = true;
					}
				}
			}
			bounds_changed = bounds_changed || changed;
		}
		while (changed);
		if (bounds_changed && iter > 0) nReset = 0;

		// check corner condition
		if (nActive + 1 >= dimension) break;

		// compute the gradient of the sub problem
		// projected onto the equality constraint
		DirLen2old = DirLen2;
		DirLen2 = 0.0;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			value = gradient(i) - sub;
			direction(i) = value;
			DirLen2 += value * value;
		}

		// update the conjugate direction
		nReset--;
		if (nReset == 0)
		{
			// In theory we are done.
			// However, we may which to continue
			// in order to avoid numerical problems.
			break;
		}
		else if (nReset < 0)
		{
			// the set of active constraints has changed
			conjugate = direction;
			nReset = dimension - 1 - nActive;
		}
		else
		{
			double beta = DirLen2 / DirLen2old;
			for (i = 0; i < dimension; i++)
			{
				if (active(i)) continue;
				else conjugate(i) = direction(i) + beta * conjugate(i);
			}
		}

		// compute the unconstrained Newton step
		first = 0.0;
		second = 0.0;
		norm2 = 0.0;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			double c = conjugate(i);
			double Qc = 0.0;
			for (j = 0; j < dimension; j++)
			{
				if (active(j)) continue;
				Qc += quadratic(i, j) * conjugate(j);
			}
			first += gradient(i) * c;
			second += c * Qc;
			norm2 += c * c;
		}
		if (second <= 0.0) break;
		step = -first / second;

		// check stopping condition
		if (nReset <= 1 && step * step * norm2 < accuracy2) break;

		// clip the step
		bound = -1;
		clipped = step;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			if (point(i) + clipped * conjugate(i) > boxMax(i))
			{
				clipped = (boxMax(i) - point(i)) / conjugate(i);
				bound = i;
			}
			else if (point(i) + clipped * conjugate(i) < boxMin(i))
			{
				clipped = (boxMin(i) - point(i)) / conjugate(i);
				bound = i;
			}
		}

		// update the gradient to avoid full recomputation
		if (5 * nActive > dimension && nUpdatesLeft > 0)
		{
			// update the gradient
			for (i = 0; i < dimension; i++)
			{
				if (active(i)) continue;
				double delta = clipped * conjugate(i);
				for (j = 0; j < dimension; j++) gradient(j) += quadratic(i, j) * delta;
			}
			nUpdatesLeft--;
		}
		else nUpdatesLeft = -1;

		// go!
		changed = false;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			double p = point(i);
			double newp = p + clipped * conjugate(i);
			point(i) = newp;
			if (p != newp) changed = true;
		}

		// stop if the step has no numerical effect
		if (! changed) break;

		// check the bound carefully
		if (bound != -1)
		{
// 			if (point(bound) > (1.0 - 1e-12) * boxMax(bound)) point(bound) = boxMax(bound);
// 			else if (point(bound) < (1.0 + 1e-12) * boxMin(bound)) point(bound) = boxMin(bound);
			double threshold = 0.5 * (boxMax(bound) + boxMin(bound));
			if (point(bound) > threshold) point(bound) = boxMax(bound);
			else point(bound) = boxMin(bound);
			nReset = 0;
		}
	}

	if (iter >= maxiter)
	{
		throw SHARKEXCEPTION("QpSvmCG did not converge");
		printf("\n\n***************** QpSvmCG did not converge!!!\n\n");
		printf("boxMin:\n");
		writeArray(boxMin, std::cout);
		printf("boxMax:\n");
		writeArray(boxMax, std::cout);
		printf("point:\n");
		writeArray(point, std::cout);
		printf("gradient:\n");
		writeArray(gradient, std::cout);
		printf("direction:\n");
		writeArray(direction, std::cout);
		printf("conjugate:\n");
		writeArray(conjugate, std::cout);
		printf("active:\n");
		writeArray(active, std::cout);
	}
}


////////////////////////////////////////////////////////////////////////////////


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
			double nominator = gradient(i) - gradient(j);
			double denominator = diagonal(i) + diagonal(j) - 2.0 * qi[j];
			double mu = nominator / denominator;
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
	SMO();

	// return alpha
	for (i = 0; i < dimension; i++)
	{
		solutionVector(permutation(i)) = alpha(i);
	}
}

void QpBoxDecomp::Continue(const Array<double>& boxLower,
						   const Array<double>& boxUpper,
						   Array<double>& solutionVector)
{
	SIZE_CHECK(boxLower.ndim() == 1);
	SIZE_CHECK(boxUpper.ndim() == 1);
	SIZE_CHECK(boxLower.dim(0) == dimension);
	SIZE_CHECK(boxUpper.dim(0) == dimension);
	solutionVector.resize(alpha.dim(0), false);

	unsigned int i, j;

	for (i = 0; i < dimension; i++)
	{
		j = permutation(i);
		boxMin(i) = boxLower(j);
		boxMax(i) = boxUpper(j);
	}

	// initial shrinking
	Shrink();

	// decomposition loop
	SMO();

	// return alpha
	for (i = 0; i < dimension; i++)
	{
		solutionVector(permutation(i)) = alpha(i);
	}
}

void QpBoxDecomp::Continue(const Array<double>& gradientUpdate,
						   Array<double>& solutionVector)
{
	SIZE_CHECK(gradientUpdate.ndim() == 1);
	SIZE_CHECK(gradientUpdate.dim(0) == dimension);
	solutionVector.resize(alpha.dim(0), false);

	unsigned int i;

	// update the gradient
	for (i = 0; i < dimension; i++)
	{
		double value = gradientUpdate(permutation(i));
		gradient(i) += value;
		linear(i) = value;
	}

	// re-compute active set
	Unshrink(false);

	// decomposition loop
	SMO();

	// return alpha
	for (i = 0; i < dimension; i++) solutionVector(permutation(i)) = alpha(i);
}

void QpBoxDecomp::SMO()
{
	unsigned int a, i;
	float* q;

	bUnshrinked = false;
	unsigned int shrinkCounter = (active < 1000) ? active : 1000;

	// decomposition loop
	iter = 0;
	optimal = false;
	while (iter != maxIter)
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

		// SMO update
		{
			double ai = alpha(i);
			double Li = boxMin(i);
			double Ui = boxMax(i);

			// update alpha, that is, solve the sub-problem defined by i
			double nominator = gradient(i);
			double denominator = diagonal(i);
			double mu = nominator / denominator;
			if (ai + mu < Li) mu = Li - ai;
			else if (ai + mu > Ui) mu = Ui - ai;
			alpha(i) += mu;

			// get the matrix rows corresponding to the working set
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
			Unshrink(true);
			bUnshrinked = true;
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


QpBoxAllInOneDecomp::QpBoxAllInOneDecomp(CachedMatrix& kernel)
		: kernelMatrix(kernel)
{
	examples = kernelMatrix.getMatrixSize();
	maxIter = -1;
}

QpBoxAllInOneDecomp::~QpBoxAllInOneDecomp()
{
}


void QpBoxAllInOneDecomp::Solve(unsigned int classes,
								const Array<double>& target,
								const Array<double>& prototypes,
								double C,
								Array<double>& solutionVector,
								double eps)
{
	SIZE_CHECK(target.ndim() == 2);

	this->classes = classes;
	variables = examples * classes;

	SIZE_CHECK(target.dim(0) == examples);
	SIZE_CHECK(target.dim(1) == 1);

	unsigned int a, b, bc, i, j;
	float* q = NULL;

	// precompute inner products of prototypes
	// and lengths of differences of prototypes
	m_prototypeProduct.resize(classes, classes, false);
	m_prototypeLength.resize(classes, classes, false);
	for (a = 0; a < classes; a++)
	{
		for (b = 0; b <= a; b++)
		{
			m_prototypeProduct(a, b) = m_prototypeProduct(b, a)
									   = scalarProduct(prototypes[a], prototypes[b]);
			Array<double> diff = prototypes[a] - prototypes[b];
			m_prototypeLength(a, b) = m_prototypeLength(b, a)
									  = sqrt(scalarProduct(diff, diff));
		}
	}

	// prepare lists
	alpha.resize(variables, false);
	diagonal.resize(examples, false);
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
		diagonal(i) = kernelMatrix.Entry(i, i);
		example[i].index = i;
		example[i].label = (unsigned int)target(i, 0);
		example[i].active = classes;
		example[i].variables = &storage[classes * i];
	}
	for (i = 0; i < variables; i++)
	{
		variable[i].example = i / classes;
		variable[i].index = i % classes;
		variable[i].label = i % classes;
		storage[i] = i;
	}

	// prepare the solver internal variables
	for (a = 0; a < examples; a++)
	{
		unsigned int y = example[a].label;
		for (b = 0; b < classes; b++)
		{
			j = example[a].variables[b];
			unsigned int v = variable[j].label;
			alpha(j) = solutionVector(j);
			linear(j) = 1.0;
			boxMin(j) = 0.0;
			if (y == v) boxMax(j) = 0.0;
			else boxMax(j) = C;
		}
	}
	gradient = linear;

	epsilon = eps;

	activeEx = examples;
	activeVar = variables;

	unsigned int e = 0xffffffff;
	for (i = 0; i < variables; i++)
	{
		if (boxMax(i) < boxMin(i)) throw SHARKEXCEPTION("[QpBoxAllInOneDecomp::Solve] The feasible region is empty.");

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
				bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					j = ex->variables[b];
					unsigned int vj = variable[j].label;
					unsigned int yj = ex->label;
					double km = Modifier(vi, vj, yi, yj);
					gradient(j) -= v * km * k;
				}
			}
		}
	}

	// initial shrinking (useful for dummy variables and warm starts)
	Shrink();

	bUnshrinked = false;
	unsigned int shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

	// decomposition loop
	iter = 0;
	optimal = false;
	while (iter != maxIter)
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
			double km = Modifier(vi, vi, yi, yi);
			double nominator = gradient(i);
			double denominator = km * diagonal(ei);
			double mu = nominator / denominator;
			if (ai + mu < Li) mu = Li - ai;
			else if (ai + mu > Ui) mu = Ui - ai;
			alpha(i) += mu;

			// get the matrix row corresponding to the working set
			q = kernelMatrix.Row(ei, 0, activeEx);

			// update the gradient
			for (a = 0; a < activeEx; a++)
			{
				double k = q[a];
				tExample* ex = &example[a];
				bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					j = ex->variables[b];

					unsigned int vj = variable[j].label;
					unsigned int yj = ex->label;
					double km = Modifier(vi, vj, yi, yj);
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
	if (iter == maxIter) optimal = false;

	// return alpha
	for (i = 0; i < variables; i++)
	{
		unsigned int j = classes * example[variable[i].example].index + variable[i].label;
		solutionVector(j) = alpha(i);
	}
}

void QpBoxAllInOneDecomp::Continue(double largerC, Array<double>& solutionVector)
{
	unsigned int a, b, bc, i = 0, j;
	float* q = NULL;

	solutionVector.resize(variables, false);

	// set the new box size
	for (a = 0; a < variables; a++)
	{
		if (boxMax(a) > 0.0) boxMax(a) = largerC;
	}

	// initial shrinking (useful for dummy variables
	// AND recent non-support-vectors)
	// Shrink();

	bUnshrinked = false;
	unsigned int shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

	// decomposition loop
	while (true)
	{
		// select a working set and check for optimality
		if (SelectWorkingSet(i))
		{
			// seems to be optimal

			// do costly unshrinking
			Unshrink(true);

			// check again on the whole problem
			if (SelectWorkingSet(i)) break;

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
			double km = Modifier(vi, vi, yi, yi);
			double nominator = gradient(i);
			double denominator = km * diagonal(ei);
			double mu = nominator / denominator;
			if (ai + mu < Li) mu = Li - ai;
			else if (ai + mu > Ui) mu = Ui - ai;
			alpha(i) += mu;

			// get the matrix row corresponding to the working set
			q = kernelMatrix.Row(ei, 0, activeEx);

			// update the gradient
			for (a = 0; a < activeEx; a++)
			{
				double k = q[a];
				tExample* ex = &example[a];
				bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					j = ex->variables[b];

					unsigned int vj = variable[j].label;
					unsigned int yj = ex->label;
					double km = Modifier(vi, vj, yi, yj);
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

	// return alpha
	for (i = 0; i < variables; i++)
	{
		unsigned int j = classes * example[variable[i].example].index + variable[i].label;
		solutionVector(j) = alpha(i);
	}
}

double QpBoxAllInOneDecomp::Modifier(unsigned int v_i, unsigned int v_j, unsigned int y_i, unsigned int y_j)
{
	double p = m_prototypeProduct(y_i, y_j) - m_prototypeProduct(v_i, y_j) - m_prototypeProduct(y_i, v_j) + m_prototypeProduct(v_i, v_j);
	// check for zero entry in sparse matrix
	if (p == 0.0) return 0.0;
	p /= (m_prototypeLength(y_i, v_i) * m_prototypeLength(y_j, v_j));
	return p;
}

bool QpBoxAllInOneDecomp::SelectWorkingSet(unsigned int& i)
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

void QpBoxAllInOneDecomp::Shrink()
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
			Unshrink(true);
			bUnshrinked = true;
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

void QpBoxAllInOneDecomp::Unshrink(bool complete)
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

			double km = Modifier(vi, va, yi, ya);
			double k = q[ea];
			gradient(a) -= km * k * v;
		}
	}

	for (i = 0; i < examples; i++) example[i].active = classes;
	activeEx = examples;
	activeVar = variables;

	if (! complete) Shrink();
}

void QpBoxAllInOneDecomp::DeactivateVariable(unsigned int v)
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

void QpBoxAllInOneDecomp::DeactivateExample(unsigned int e)
{
	unsigned int j = activeEx - 1;

	XCHG_A(double, diagonal, e, j);
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


QpCrammerSingerDecomp::QpCrammerSingerDecomp(CachedMatrix& kernel, const Array<double>& y, unsigned int classes)
		: kernelMatrix(kernel)
{
	SIZE_CHECK(y.ndim() == 2);

	examples = kernelMatrix.getMatrixSize();
	this->classes = classes;
	variables = examples * classes;

	SIZE_CHECK(y.dim(0) == examples);
	SIZE_CHECK(y.dim(1) == 1);

	// prepare lists
	alpha.resize(variables, false);
	diagonal.resize(examples, false);
	gradient.resize(variables, false);
	linear.resize(variables, false);
	boxMin.resize(variables, false);
	boxMax.resize(variables, false);
	example.resize(examples);
	variable.resize(variables);
	storage.resize(variables);

	// prepare the lists
	unsigned int i;
	for (i = 0; i < examples; i++)
	{
		diagonal(i) = kernelMatrix.Entry(i, i);
		example[i].index = i;
		example[i].label = (unsigned int)y(i, 0);
		example[i].active = classes;
		example[i].variables = &storage[classes * i];
	}
	for (i = 0; i < variables; i++)
	{
		variable[i].example = i / classes;
		variable[i].index = i % classes;
		variable[i].label = i % classes;
		storage[i] = i;
	}

	maxIter = -1;
}

QpCrammerSingerDecomp::~QpCrammerSingerDecomp()
{
}


void QpCrammerSingerDecomp::Solve(Array<double>& solutionVector,
								  double beta,
								  double eps)
{
	unsigned int a, b, bc, i = 0, j = 0;
	float* q = NULL;

	// prepare the solver internal variables
	for (a = 0; a < examples; a++)
	{
		unsigned int y = example[a].label;
		for (b = 0; b < classes; b++)
		{
			j = example[a].variables[b];
			alpha(j) = solutionVector(j);
			unsigned int v = variable[j].label;
			if (y == v)
			{
				linear(j) = beta;
				boxMin(j) = -1e100;
				boxMax(j) = 1.0;
			}
			else
			{
				linear(j) = 0.0;
				boxMin(j) = -1e100;
				boxMax(j) = 0.0;
			}
		}
	}

	epsilon = eps;

	activeEx = examples;
	activeVar = variables;

	// compute the initial gradient
	gradient = linear;
	unsigned int e = 0xffffffff;
	for (a = 0; a < activeEx; a++)
	{
		tExample* ex = &example[a];
		bc = ex->active;
		for (b = 0; b < bc; b++)
		{
			j = ex->variables[b];
			double v = alpha(j);
			if (v != 0.0)
			{
				unsigned int h;
				unsigned int vj = variable[j].label;
				if (e != a)
				{
					q = kernelMatrix.Row(a, 0, activeEx);
					e = a;
				}
				for (h = 0; h < activeVar; h++)
				{
					unsigned int vh = variable[h].label;
					if (vh == vj)
					{
						unsigned int eh = variable[h].example;
						gradient(h) -= v * q[eh];
					}
				}
			}
		}
	}

	// initial shrinking
	Shrink();

	bUnshrinked = false;
	unsigned int shrinkCounter = (activeVar < 1000) ? activeVar : 1000;

	// decomposition loop
	iter = 0;
	optimal = false;
	while (iter != maxIter)
	{
		// select a working set and check for optimality
		if (SelectWorkingSet(i, j) <= epsilon)
		{
			// seems to be optimal

			// do costly unshrinking
			Unshrink(true);

			// check again on the whole problem
			if (SelectWorkingSet(i, j) <= epsilon)
			{
				optimal = true;
				break;
			}

			// shrink again
			Shrink();
			shrinkCounter = (activeVar < 1000) ? activeVar : 1000;
		}

		// SMO update
		{
			// there is only one example corresponding to both variables:
			unsigned int e = variable[i].example;

			double ai = alpha(i);
			double Li = boxMin(i);
			double Ui = boxMax(i);
			double aj = alpha(j);
			double Lj = boxMin(j);
			double Uj = boxMax(j);

			unsigned int vi = variable[i].label;		// these are
			unsigned int vj = variable[j].label;		// different!

			// get the matrix rows corresponding to the working set
			q = kernelMatrix.Row(e, 0, activeEx);

			// update alpha, that is, solve the sub-problem defined by i and j
			double nominator = gradient(i) - gradient(j);
			double denominator = 2.0 * diagonal(e);
			double mu = nominator / denominator;
			if (ai + mu < Li) mu = Li - ai;
			else if (ai + mu > Ui) mu = Ui - ai;
			if (aj - mu < Lj) mu = aj - Lj;
			else if (aj - mu > Uj) mu = aj - Uj;
			alpha(i) += mu;
			alpha(j) -= mu;

			// update the gradient
			for (a = 0; a < activeEx; a++)
			{
				double k = q[a];
				tExample* ex = &example[a];
				unsigned int b, bc = ex->active;
				for (b = 0; b < bc; b++)
				{
					unsigned int h = ex->variables[b];
					unsigned int vh = variable[h].label;
					if (vh == vi) gradient(h) -= mu * k;
					else if (vh == vj) gradient(h) += mu * k;
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

// Select a working set (i, j) composed of an "up"-component i and
// a "down"-component j. Return the corresponding KKT violation.
double QpCrammerSingerDecomp::SelectWorkingSet(unsigned int& i, unsigned int& j)
{
	double largest = 0.0;

	unsigned int a, b, bc, w;
	for (a = 0; a < activeEx; a++)
	{
		tExample* ex = &example[a];
		bc = ex->active;
		double m = 1e100;
		double M = -1e100;
		unsigned int n = 0, N = 1;
		for (b = 0; b < bc; b++)
		{
			w = ex->variables[b];
			double v = alpha(w);
			double g = gradient(w);
			if (v < boxMax(w))
			{
				if (g > M)
				{
					M = g;
					N = w;
				}
			}
			if (v > boxMin(w))
			{
				if (g < m)
				{
					m = g;
					n = w;
				}
			}
		}
		if (M - m > largest)
		{
			largest = M - m;
			i = N;
			j = n;
		}
	}

	return largest;
}

void QpCrammerSingerDecomp::Shrink()
{
	if (! bUnshrinked)
	{
		unsigned int i, j;
		double violation = SelectWorkingSet(i, j);
		if (violation < 10.0 * epsilon)
		{
			// unshrink the problem at this accuracy level
			Unshrink(true);
			bUnshrinked = true;
		}
	}

	// loop through the examples
	bool se = false;
	int a;
	for (a = activeEx - 1; a >= 0; a--)
	{
		// loop through the variables corresponding to the example
		tExample* ex = &example[a];
		unsigned int b, bc = ex->active;
		double m = 1e100;
		double M = -1e100;
		for (b = 0; b < bc; b++)
		{
			unsigned int w = ex->variables[b];
			double v = alpha(w);
			double g = gradient(w);
			if (v < boxMax(w))
			{
				if (g > M)
				{
					M = g;
				}
			}
			if (v > boxMin(w))
			{
				if (g < m)
				{
					m = g;
				}
			}
		}
		for (b = 0; b < bc; b++)
		{
			unsigned int w = ex->variables[b];
			double v = alpha(w);
			double g = gradient(w);
			if (v == boxMin(w) && g < m)
			{
				DeactivateVariable(w);
				b--;
				bc--;
			}
			else if (v == boxMax(w) && g > M)
			{
				DeactivateVariable(w);
				b--;
				bc--;
			}
		}
		if (bc == 0)
		{
			DeactivateExample(a);
			se = true;
		}
	}

	if (se)
	{
		// shrink the corresponding cache entries
		for (a = 0; a < (int)activeEx; a++)
		{
			if (kernelMatrix.getCacheRowSize(a) > activeEx) kernelMatrix.CacheRowResize(a, activeEx);
		}
	}
}

void QpCrammerSingerDecomp::Unshrink(bool complete)
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
		q = kernelMatrix.Row(ei, 0, examples, true);
		for (a = activeVar; a < variables; a++)
		{
			if (variable[a].label == vi)
			{
				unsigned int ea = variable[a].example;
				gradient(a) -= q[ea] * v;
			}
		}
	}

	for (i = 0; i < examples; i++) example[i].active = classes;
	activeEx = examples;
	activeVar = variables;

	if (! complete) Shrink();
}

void QpCrammerSingerDecomp::DeactivateVariable(unsigned int v)
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

void QpCrammerSingerDecomp::DeactivateExample(unsigned int e)
{
	unsigned int j = activeEx - 1;

	XCHG_A(double, diagonal, e, j);
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


////////////////////////////////////////////////////////////////////////////////


void QpBoxAndEqCG::Solve(const Array<double>& quadratic,
						 const Array<double>& linear,
						 const Array<double>& boxMin,
						 const Array<double>& boxMax,
						 const Array<double>& eqMat,
						 Array<double>& point,
						 double accuracy)
{
	SIZE_CHECK(quadratic.ndim() == 2);
	SIZE_CHECK(linear.ndim() == 1);
	SIZE_CHECK(boxMin.ndim() == 1);
	SIZE_CHECK(boxMax.ndim() == 1);
	SIZE_CHECK(eqMat.ndim() == 2);
	SIZE_CHECK(point.ndim() == 1);

	int dimension = quadratic.dim(0);
	int equations = eqMat.dim(0);

	SIZE_CHECK((int)quadratic.dim(1) == dimension);
	SIZE_CHECK((int)linear.dim(0) == dimension);
	SIZE_CHECK((int)boxMin.dim(0) == dimension);
	SIZE_CHECK((int)boxMax.dim(0) == dimension);
	SIZE_CHECK((int)eqMat.dim(1) == dimension);
	SIZE_CHECK((int)point.dim(0) == dimension);

	int i, j;
	Array<double> direction(dimension);
	Array<bool> active(dimension);
	Array<double> gradient(dimension);
	Array<double> conjugate(dimension);
	Array<double> equality(eqMat);
	conjugate = 0.0;
	double DirLen2 = 0.0;
	double DirLen2old = 0.0;
	double accuracy2 = accuracy * accuracy;
	bool changed, bounds_changed;
	int nActive = 0;
	int nUpdatesLeft = -1;
	int bound = -1;
	int nReset = 0;
	double step = 0.0;
	double clipped = 0.0;
	double value;
	double first = 0.0;
	double second = 0.0;
	double norm2;

	active = false;

	int iter, maxiter = 10 * dimension * dimension;
	for (iter = 0; iter < maxiter; iter++)
	{
		if (nUpdatesLeft == -1)
		{
			// compute the gradient from scratch
			for (i = 0; i < dimension; i++)
			{
				value = linear(i);
				for (j = 0; j < dimension; j++) value += quadratic(i, j) * point(j);
				gradient(i) = value;
			}
			nUpdatesLeft = 100;
		}

		// compute the set of active inequality constraints
		bounds_changed = false;
		do
		{
			changed = false;
			for (i = 0; i < dimension; i++)
			{
				if (active(i))
				{
					equality = eqMat;
					active(i) = false;
					Project(active, equality, gradient, direction);
					double g = direction(i);
					if ((g <= 0.0 && point(i) == boxMin(i))
							|| (g >= 0.0 && point(i) == boxMax(i)))
					{
						nActive--;
						changed = true;
					}
					else active(i) = true;
				}
				else
				{
					equality = eqMat;
					active(i) = true;
					Project(active, equality, gradient, direction);
					double g = direction(i);
					if ((g > 0.0 && point(i) == boxMin(i))
							|| (g < 0.0 && point(i) == boxMax(i)))
					{
						nActive++;
						changed = true;
					}
					else active(i) = false;
				}
			}
			bounds_changed = bounds_changed || changed;
		}
		while (changed);
		if (bounds_changed && iter > 0) nReset = 0;

		// check corner condition
		if (nActive + 1 >= dimension) break;

		// compute the gradient of the sub problem
		// projected onto the equality constraint
		equality = eqMat;
		Project(active, equality, gradient, direction);

		// compute the length of the remaining vector
		DirLen2old = DirLen2;
		DirLen2 = 0.0;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			value = direction(i);
			DirLen2 += value * value;
		}

		// update the conjugate direction
		nReset--;
		if (nReset == 0)
		{
			// In theory we are done.
			// However, we may which to continue
			// in order to avoid numerical problems.
			break;
		}
		else if (nReset < 0)
		{
			// the set of active constraints has changed
			conjugate = direction;
			nReset = dimension - equations - nActive;
		}
		else
		{
			double beta = DirLen2 / DirLen2old;
			for (i = 0; i < dimension; i++)
			{
				if (active(i)) continue;
				else conjugate(i) = direction(i) + beta * conjugate(i);
			}
		}

		// compute the unconstrained Newton step
		first = 0.0;
		second = 0.0;
		norm2 = 0.0;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			double c = conjugate(i);
			double Qc = 0.0;
			for (j = 0; j < dimension; j++)
			{
				if (active(j)) continue;
				Qc += quadratic(i, j) * conjugate(j);
			}
			first += gradient(i) * c;
			second += c * Qc;
			norm2 += c * c;
		}
		if (second <= 0.0) break;
		step = -first / second;

		// check stopping condition
		if (nReset <= 1 && step * step * norm2 < accuracy2) break;

		// clip the step
		bound = -1;
		clipped = step;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			if (point(i) + clipped * conjugate(i) > boxMax(i))
			{
				clipped = (boxMax(i) - point(i)) / conjugate(i);
				bound = i;
			}
			else if (point(i) + clipped * conjugate(i) < boxMin(i))
			{
				clipped = (boxMin(i) - point(i)) / conjugate(i);
				bound = i;
			}
		}

		// update the gradient to avoid full recomputation
		if (5 * nActive > dimension && nUpdatesLeft > 0)
		{
			// update the gradient
			for (i = 0; i < dimension; i++)
			{
				if (active(i)) continue;
				double delta = clipped * conjugate(i);
				for (j = 0; j < dimension; j++) gradient(j) += quadratic(i, j) * delta;
			}
			nUpdatesLeft--;
		}
		else nUpdatesLeft = -1;

		// go!
		changed = false;
		for (i = 0; i < dimension; i++)
		{
			if (active(i)) continue;
			double p = point(i);
			double newp = p + clipped * conjugate(i);
			point(i) = newp;
			if (p != newp) changed = true;
		}

		// stop if the step has no numerical effect
		if (! changed) break;

		// check the bound carefully
		if (bound != -1)
		{
// 			if (point(bound) > (1.0 - 1e-12) * boxMax(bound)) point(bound) = boxMax(bound);
// 			else if (point(bound) < (1.0 + 1e-12) * boxMin(bound)) point(bound) = boxMin(bound);
			double threshold = 0.5 * (boxMax(bound) + boxMin(bound));
			if (point(bound) > threshold) point(bound) = boxMax(bound);
			else point(bound) = boxMin(bound);
			nReset = 0;
		}
	}

	if (iter >= maxiter)
	{
		throw SHARKEXCEPTION("QpBoxAndEqCG did not converge");
		printf("\n\n***************** QpBoxAndEqCG did not converge!!!\n\n");
		printf("boxMin:\n");
		writeArray(boxMin, std::cout);
		printf("boxMax:\n");
		writeArray(boxMax, std::cout);
		printf("point:\n");
		writeArray(point, std::cout);
		printf("gradient:\n");
		writeArray(gradient, std::cout);
		printf("direction:\n");
		writeArray(direction, std::cout);
		printf("conjugate:\n");
		writeArray(conjugate, std::cout);
		printf("active:\n");
		writeArray(active, std::cout);
	}
}

void QpBoxAndEqCG::Orthogonalize(int oc, const Array<double>& ortho, const Array<bool>& active, ArrayReference<double> vec)
{
	int o, d, dim = ortho.dim(1);
	double scp, norm;

	// orthogonalize
	for (o = 0; o < oc; o++)
	{
		scp = 0.0;
		for (d = 0; d < dim; d++)
		{
			if (active(d)) continue;
			scp += ortho(o, d) * vec(d);
		}
		for (d = 0; d < dim; d++)
		{
			if (active(d)) vec(d) = 0.0;
			else vec(d) -= scp * ortho(o, d);
		}
	}

	// normalize
	scp = 0.0;
	for (d = 0; d < dim; d++)
	{
		if (active(d)) continue;
		scp += vec(d) * vec(d);
	}
	norm = sqrt(scp);
	for (d = 0; d < dim; d++)
	{
		if (active(d)) continue;
		vec(d) /= norm;
	}
}

void QpBoxAndEqCG::Orthogonalize(Array<double>& eq, const Array<bool>& active)
{
	int e, ec = eq.dim(0);
	for (e = 0; e < ec; e++)
	{
		Orthogonalize(e, eq, active, eq[e]);
	}
}

void QpBoxAndEqCG::Project(const Array<bool>& active, Array<double>& eq, const Array<double>& gradient, Array<double>& direction)
{
	Orthogonalize(eq, active);

	int o, oc = eq.dim(0);
	int d, dim = eq.dim(1);
	double scp;
	for (o = 0; o < oc; o++)
	{
		scp = 0.0;
		for (d = 0; d < dim; d++)
		{
			if (active(d)) continue;
			scp += eq(o, d) * gradient(d);
		}
		for (d = 0; d < dim; d++)
		{
			if (active(d)) direction(d) = 0.0;
			else direction(d) = gradient(d) - scp * eq(o, d);
		}
	}
}


////////////////////////////////////////////////////////////////////////////////


void QpBoxAndEqDecomp::Solve(const Array<double>& quadratic,
							 const Array<double>& linear,
							 const Array<double>& boxMin,
							 const Array<double>& boxMax,
							 const Array<double>& eqMat,
							 Array<double>& point,
							 double accuracy)
{
	SIZE_CHECK(quadratic.ndim() == 2);
	SIZE_CHECK(linear.ndim() == 1);
	SIZE_CHECK(boxMin.ndim() == 1);
	SIZE_CHECK(boxMax.ndim() == 1);
	SIZE_CHECK(eqMat.ndim() == 2);
	SIZE_CHECK(point.ndim() == 1);

	int dimension = quadratic.dim(0);
	int equations = eqMat.dim(0);

	SIZE_CHECK((int)quadratic.dim(1) == dimension);
	SIZE_CHECK((int)linear.dim(0) == dimension);
	SIZE_CHECK((int)boxMin.dim(0) == dimension);
	SIZE_CHECK((int)boxMax.dim(0) == dimension);
	SIZE_CHECK((int)eqMat.dim(1) == dimension);
	SIZE_CHECK((int)point.dim(0) == dimension);

	Array<double> eq(eqMat);
	Orthogonalize(eq);

	Array<double> gradient(linear);
	Array<double> direction(dimension);
	int workingsize = 2 * equations + 1;
	int i, j, k;

	Array<int> workingset(workingsize);
	Array<double> subQ(workingsize, workingsize);
	Array<double> subL(workingsize);
	Array<double> subBoxMin(workingsize);
	Array<double> subBoxMax(workingsize);
	Array<double> subEqMat(equations, workingsize);
	Array<double> subPoint(workingsize);

	// initialize the gradient
	for (i = 0; i < dimension; i++)
	{
		double p = point(i);
		if (p != 0.0)
		{
			for (j = 0; j < dimension; j++) gradient(j) += p * quadratic(i, j);
		}
	}

	// decomposition loop
	while (true)
	{
		// select a working set
		double gNorm1 = 0.0;
		{
			// project the gradient onto the equality constraints
			Project(eq, gradient, direction);

			// find the k largest free components of the direction
			for (i = 0; i < workingsize; i++)
			{
				int best = 0;
				double bestG = 0.0;
				for (j = 0; j < dimension; j++)
				{
					double d = direction(j);
					if (point(j) == boxMin(j) && d >= 0.0) continue;
					if (point(j) == boxMax(j) && d <= 0.0) continue;
					d = fabs(d);
					if (d > bestG)
					{
						for (k = 0; k < i; k++) if (workingset(k) == j) break;
						if (k == i)
						{
							bestG = d;
							best = j;
						}
					}
				}
				workingset(i) = best;
				gNorm1 += bestG;
			}
		}

		// check the stopping condition
		if (gNorm1 < accuracy) break;

		// define the sub problem and solve it
		for (i = 0; i < workingsize; i++)
		{
			j = workingset(i);
			for (k = 0; k < workingsize; k++) subQ(i, k) = quadratic(j, workingset(k));
			subL(i) = linear(j);
			subBoxMin(i) = boxMin(j);
			subBoxMax(i) = boxMax(j);
			for (k = 0; k < equations; k++) subEqMat(k, i) = eqMat(k, j);
			subPoint(i) = point(j);
		}
		QpBoxAndEqCG::Solve(subQ, subL, subBoxMin, subBoxMax, subEqMat, subPoint);

		// update point and gradient
		for (i = 0; i < workingsize; i++)
		{
			j = workingset(i);
			double delta = subPoint(i) - point(j);
			if (delta == 0.0) continue;
			for (k = 0; k < dimension; k++) gradient(k) += delta * quadratic(j, k);
			point(j) = subPoint(i);
		}
	}
}

void QpBoxAndEqDecomp::Orthogonalize(Array<double>& eq)
{
	int e, ec = eq.dim(0);
	int o, d, dim = eq.dim(1);
	double scp, norm;
	for (e = 0; e < ec; e++)
	{
		// orthogonalize
		for (o = 0; o < e; o++)
		{
			scp = 0.0;
			for (d = 0; d < dim; d++)
			{
				scp += eq(o, d) * eq(e, d);
			}
			for (d = 0; d < dim; d++)
			{
				eq(e, d) -= scp * eq(o, d);
			}
		}

		// normalize
		scp = 0.0;
		for (d = 0; d < dim; d++)
		{
			scp += eq(e, d) * eq(e, d);
		}
		norm = sqrt(scp);
		for (d = 0; d < dim; d++)
		{
			eq(e, d) /= norm;
		}
	}
}

void QpBoxAndEqDecomp::Project(const Array<double>& eq, const Array<double>& gradient, Array<double>& direction)
{
	int o, oc = eq.dim(0);
	int d, dim = eq.dim(1);
	double scp;
	for (o = 0; o < oc; o++)
	{
		scp = 0.0;
		for (d = 0; d < dim; d++)
		{
			scp += eq(o, d) * gradient(d);
		}
		for (d = 0; d < dim; d++)
		{
			direction(d) = gradient(d) - scp * eq(o, d);
		}
	}
}


////////////////////////////////////////////////////////////


// #define ACCURACY 1e-12
//
//
// //
// // perform an affine linear transformation
// // of the given problem with a twofold
// // purpose:
// //
// // (1) reduce the dimension according to
// //     the equality constraints
// // (2) make the objective functions
// //     isotropic
// //
// QpAffine::QpAffine(
// 			const Array<double>& quadratic,
// 			const Array<double>& linear,
// 			const Array<double>& eqMat,
// 			const Array<double>& ineqMat,
// 			const Array<double>& ineqVec
// 	  )
// : quadraticOrig(quadratic)
// , constraintMatOrig(ineqMat)
// , constraintVecOrig(ineqVec)
// {
// 	SIZE_CHECK(quadratic.ndim() == 2);
// 	SIZE_CHECK(linear.ndim() == 1);
// 	SIZE_CHECK(eqMat.ndim() == 2);
// 	SIZE_CHECK(ineqMat.ndim() == 2);
// 	SIZE_CHECK(ineqVec.ndim() == 1);
//
// 	dimension = quadratic.dim(0);
//
// 	SIZE_CHECK((int)quadratic.dim(1) == dimension);
// 	SIZE_CHECK((int)linear.dim(0) == dimension);
// 	SIZE_CHECK((int)eqMat.dim(1) == dimension);
// 	SIZE_CHECK((int)ineqMat.dim(1) == dimension);
// 	SIZE_CHECK(ineqMat.dim(0) == ineqVec.dim(0));
//
// 	int i, j, k;
// 	int c, cc;
// 	std::vector<double*> list;
//
// 	// compute an orthogonal transformation
// 	// s.t. the last k coordinates are within
// 	// the equality constraint normal space
// 	Array<double> trans1(dimension, dimension);
// 	Array<double> inverse1(dimension, dimension);
// 	cc = eqMat.dim(0);
// 	for (i=dimension-1, c=0; c<cc; c++)
// 	{
// 		for (j=0; j<dimension; j++) trans1(i, j) = eqMat(c, j);
// 		Orthogonalize(list, trans1[i]);
// 		if (Normalize(trans1[i]))
// 		{
// 			list.push_back(&trans1(i, 0));
// 			i--;
// 		}
// 		if ((int)list.size() >= dimension) throw SHARKEXCEPTION("[QpAffine::QpAffine] too many equality constraints");
// 	}
// 	freedim = i + 1;
// 	for (; i >= 0; i--)
// 	{
// 		do
// 		{
// 			for (j=0; j<dimension; j++) trans1(i, j) = Rng::gauss();
// 			Orthogonalize(list, trans1[i]);
// 		}
// 		while (! Normalize(trans1[i]));
// 		list.push_back(&trans1(i, 0));
// 	}
// 	inverse1 = trans1;
// 	inverse1.transpose();
//
// 	// compute a positive definite transformation
// 	// s.t. the quadratic term vanishes on the
// 	// last k components and becomes the unit
// 	// matrix in the remaining components.
// 	Array<double> trans2(freedim, freedim);
// 	Array<double> inverse2(freedim, freedim);
// 	Array2D<double> tmp(freedim, dimension);
// 	Array2D<double> A(freedim, freedim);
// 	Array2D<double> U(freedim, freedim);
// 	Array2D<double> V(freedim, freedim);
// 	Array<double> lambda(freedim);
// 	MatMul(trans1, quadratic, tmp);
// 	MatMul(tmp, inverse1, A);
// 	svd(A, U, V, lambda);
// 	for (k=0; k<freedim; k++) lambda(k) = sqrt(lambda(k));
// 	V.transpose();
// 	for (i=0; i<freedim; i++)
// 	{
// 		for (j=0; j<freedim; j++)
// 		{
// 			double value;
// 			value = 0.0;
// 			for (k=0; k<freedim; k++) value += U(i, k) / lambda(k) * V(k, j);
// 			trans2(i, j) = value;
// 			value = 0.0;
// 			for (k=0; k<freedim; k++) value += U(i, k) * lambda(k) * V(k, j);
// 			inverse2(i, j) = value;
// 		}
// 	}
//
// 	// compute the composed transformation,
// 	// its inverse and its transpose
// 	transform.resize(freedim, dimension, false);
// 	inverse.resize(dimension, freedim, false);
// 	MatMul(trans2, trans1, transform);
// 	MatMul(inverse1, inverse2, inverse);
//
// 	// transform the objective function and
// 	// the inequality constraints
// 	this->linear.resize(freedim, false);
// 	MatVec(transform, linear, this->linear);
//
// 	cc = ineqMat.dim(0);
// 	constraintMatTrans.resize(cc, freedim, false);
// 	for (c=0; c<cc; c++) MatVec(transform, ineqMat[c], constraintMatTrans[c]);
// }
//
// QpAffine::~QpAffine()
// {
// }
//
//
// void QpAffine::Solve(Array<double>& point)
// {
// 	// strategy:
// 	// transform point
// 	// loop
// 	//   compute direction
// 	//   determine active constraints
// 	//   transform active constraints into an orthogonal system
// 	//   project direction onto constraints
// 	//   compute the newton step
// 	//   check when we bump into the next inactive constraint
// 	//   check if we can reach the optimum, then move there and break
// 	//   move to the collision position
// 	// end loop
// 	// retransform point
//
// 	SIZE_CHECK(point.ndim() == 1);
// 	SIZE_CHECK((int)point.dim(0) == dimension);
//
// 	int c, cc = constraintVecOrig.dim(0);
// 	int a;
// 	int i, j;
// 	bool changed;
// 	Array<bool> active(cc);
// 	std::vector<double*> list;
// 	std::vector<double*> single(1);
// 	Array<double> ortho(cc, freedim);
// 	Array<double> direction(freedim);
// 	Array<double> pt(freedim);
// 	Array<double> constraintVecTrans(cc);
// 	Array<double> linearTrans(freedim);
//
// 	// compute the linear part
// 	Array<double> tmp1(dimension);
// 	Array<double> tmp2(freedim);
// 	MatVec(quadraticOrig, point, tmp1);
// 	MatVec(transform, tmp1, tmp2);
// 	for (i=0; i<freedim; i++) linearTrans(i) = linear(i) + tmp2(i);
//
// 	// compute the inequality constraint vector
// 	for (c=0; c<cc; c++)
// 	{
// 		double value = constraintVecOrig(c);
// 		for (i=0; i<dimension; i++) value += constraintMatOrig(c, i) * point(i);
// 		constraintVecTrans(c) = value;
// 	}
//
// 	// transform the point
// 	pt = 0.0;
//
// 	// loop
// 	while (true)
// 	{
// 		list.clear();
// 		a = 0;
//
// 		for (i=0; i<freedim; i++) direction(i) = -(linearTrans(i) + pt(i));
//
// 		active = false;
// 		do
// 		{
// 			changed = false;
// 			for (c=0; c<cc; c++)
// 			{
// 				if (active(c)) continue;
// 				double scp = Scp(direction, constraintMatTrans[c]);
// 				double value = Scp(pt, constraintMatTrans[c]) + constraintVecTrans(c);
// 				if (scp > 0.0 && value >= -ACCURACY)
// 				{
// 					active(c) = true;
// 					changed = true;
// 					for (i=0; i<freedim; i++) ortho(a, i) = constraintMatTrans(c, i);
// 					Orthogonalize(list, ortho[a]);
// 					if (Normalize(ortho[a]))
// 					{
// 						list.push_back(&ortho(a, 0));
// 						single[0] = &ortho(a, 0);
// 						Orthogonalize(single, direction);
// 						a++;
// 					}
// 				}
// 			}
// 		}
// 		while (changed);
//
// 		if (! Normalize(direction)) break;
//
// 		// compute the newton step
// 		double step = 0.0;
// 		for (i=0; i<freedim; i++)
// 		{
// 			step -= (linearTrans(i) + pt(i)) * direction(i);
// 		}
// 		if (fabs(step) < ACCURACY) break;
//
// 		// check inactive constraints
// 		double bestDist = step;
// 		for (c=0; c<cc; c++)
// 		{
// 			if (active(c)) continue;
// 			double scp = Scp(direction, constraintMatTrans[c]);
// 			if (scp <= 0.0) continue;
// 			double t = -(constraintVecTrans(c) + Scp(pt, constraintMatTrans[c])) / scp;
// 			if (t < bestDist) bestDist = t;
// 		}
//
// 		// move
// 		for (i=0; i<freedim; i++) pt(i) += bestDist * direction(i);
//
// 		// check stopping condition
// 		if (bestDist == step) break;
// 	}
//
// 	// transform the point back
// 	for (j=0; j<dimension; j++)
// 	{
// 		double value = 0.0;
// 		for (i=0; i<freedim; i++) value += transform(i, j) * pt(i);
// 		point(j) += value;
// 	}
// }
//
// // static
// void QpAffine::Orthogonalize(const std::vector<double*>& ortho, Array<double>& vec)
// {
// 	int i, dim = vec.dim(0);
// 	int e, ec = ortho.size();
//
// 	// orthogonalize
// 	for (e=0; e<ec; e++)
// 	{
// 		double* o = ortho[e];
// 		double scp = 0.0;
// 		for (i=0; i<dim; i++) scp += o[i] * vec(i);
// 		for (i=0; i<dim; i++) vec(i) -= scp * o[i];
// 	}
// }
//
// // static
// void QpAffine::Orthogonalize(const std::vector<double*>& ortho, ArrayReference<double> vec)
// {
// 	int i, dim = vec.dim(0);
// 	int e, ec = ortho.size();
//
// 	// orthogonalize
// 	for (e=0; e<ec; e++)
// 	{
// 		double* o = ortho[e];
// 		double scp = 0.0;
// 		for (i=0; i<dim; i++) scp += o[i] * vec(i);
// 		for (i=0; i<dim; i++) vec(i) -= scp * o[i];
// 	}
// }
//
// // static
// bool QpAffine::Normalize(Array<double>& vec)
// {
// 	// normalize
// 	double len2 = 0.0;
// 	int i, dim = vec.dim(0);
// 	for (i=0; i<dim; i++)
// 	{
// 		double entry = vec(i);
// 		len2 += entry * entry;
// 	}
// 	if (len2 == 0.0) return false;
// 	double len = sqrt(len2);
// 	for (i=0; i<dim; i++) vec(i) /= len;
//
// 	return true;
// }
//
// // static
// bool QpAffine::Normalize(ArrayReference<double> vec)
// {
// 	// normalize
// 	double len2 = 0.0;
// 	int i, dim = vec.dim(0);
// 	for (i=0; i<dim; i++)
// 	{
// 		double entry = vec(i);
// 		len2 += entry * entry;
// 	}
// 	if (len2 == 0.0) return false;
// 	double len = sqrt(len2);
// 	for (i=0; i<dim; i++) vec(i) /= len;
//
// 	return true;
// }
//
// // static
// void QpAffine::MatMul(const Array<double>& M1, const Array<double>& M2, Array<double>& result)
// {
// 	SIZE_CHECK(M1.ndim() == 2);
// 	SIZE_CHECK(M2.ndim() == 2);
// 	SIZE_CHECK(result.ndim() == 2);
//
// 	int i, j, k;
// 	int dx = result.dim(1);
// 	int dy = result.dim(0);
// 	int d = (M1.dim(1) < M2.dim(0)) ? M1.dim(1) : M2.dim(0);
// 	for (i=0; i<dy; i++)
// 	{
// 		for (j=0; j<dx; j++)
// 		{
// 			double value = 0.0;
// 			for (k=0; k<d; k++) value += M1(i, k) * M2(k, j);
// 			result(i, j) = value;
// 		}
// 	}
// }
//
// // static
// void QpAffine::MatVec(const Array<double>& Mat, const Array<double>& vec, Array<double>& result)
// {
// 	SIZE_CHECK(Mat.ndim() == 2);
// 	SIZE_CHECK(vec.ndim() == 1);
// 	SIZE_CHECK(result.ndim() == 1);
// 	SIZE_CHECK(Mat.dim(1) == vec.dim(0));
// 	SIZE_CHECK(Mat.dim(0) == result.dim(0));
//
// 	int v, vc = vec.dim(0);
// 	int r, rc = result.dim(0);
//
// 	for (r=0; r<rc; r++)
// 	{
// 		double value = 0.0;
// 		for (v=0; v<vc; v++) value += Mat(r, v) * vec(v);
// 		result(r) = value;
// 	}
// }
//
// static
// void QpAffine::MatVec(const Array<double>& Mat, const ArrayReference<double> vec, ArrayReference<double> result)
// {
// 	SIZE_CHECK(Mat.ndim() == 2);
// 	SIZE_CHECK(vec.ndim() == 1);
// 	SIZE_CHECK(result.ndim() == 1);
// 	SIZE_CHECK(Mat.dim(1) == vec.dim(0));
// 	SIZE_CHECK(Mat.dim(0) == result.dim(0));
//
// 	int v, vc = vec.dim(0);
// 	int r, rc = result.dim(0);
//
// 	for (r=0; r<rc; r++)
// 	{
// 		double value = 0.0;
// 		for (v=0; v<vc; v++) value += Mat(r, v) * vec(v);
// 		result(r) = value;
// 	}
// }
//
// // static
// double QpAffine::Scp(const Array<double>& vec1, ArrayReference<double> vec2)
// {
// 	SIZE_CHECK(vec1.ndim() == 1);
// 	SIZE_CHECK(vec2.ndim() == 1);
// 	SIZE_CHECK(vec1.dim(0) == vec2.dim(0));
//
// 	double ret = 0.0;
// 	int i, ic = vec1.dim(0);
// 	for (i=0; i<ic; i++)
// 	{
// 		ret += vec1(i) * vec2(i);
// 	}
// 	return ret;
// }
