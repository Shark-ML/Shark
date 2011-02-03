//===========================================================================
/*!
 *  \file KernelMatrices.cpp
 *
 *  \brief Quadratic kernel matrices and related interfaces
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
	accessCount = 0;
}

QPMatrix::~QPMatrix()
{
}


////////////////////////////////////////////////////////////////////////////////


KernelMatrix::KernelMatrix(KernelFunction* kernelfunction,
						   const Array<double>& data,
						   bool count )
: QPMatrix(data.dim(0))
, kernel(kernelfunction)
, countAccess( count )
{
	x.resize(matrixsize, false);
	unsigned int i;
	for (i = 0; i < matrixsize; i++)
	{
		x(i) = new ArrayReference<double>(data[i]);
	}
	countAccess = count;
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
	if (countAccess) accessCount++;
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


float* CachedMatrix::Row(unsigned int k, unsigned int begin, unsigned int end, bool temp, bool keep)
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
		if (cacheSize + end > cacheMaxSize + l) //only keep dangling entries as long as cache is not full
			keep = false;
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
			if ( !keep || l < end ) //potentially keep dangling kernel row entries
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
				// use missing entry in column vector, or calculate
				if ( cacheEntry[col].length > (int)k && k!=col )
					*p = cacheEntry[col].data[k];
				else
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
	if (t != -2)
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
	if (cacheEntry[var].data == NULL) throw SHARKEXCEPTION("[CachedMatrix::cacheAdd] out of memory error");
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

