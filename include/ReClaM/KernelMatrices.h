//===========================================================================
/*!
 *  \file KernelMatrices.h
 *
 *  \brief Quadratic kernel matrices and related interfaces
 *
 *  \author  T. Glasmachers
 *  \date	2007
 *
 *  \par Copyright (c) 1999-2007:
 *	  Institut f&uuml;r Neuroinformatik<BR>
 *	  Ruhr-Universit&auml;t Bochum<BR>
 *	  D-44780 Bochum, Germany<BR>
 *	  Phone: +49-234-32-25558<BR>
 *	  Fax:   +49-234-32-14209<BR>
 *	  eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *	  www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *  <BR>
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


#ifndef _KernelMatrices_H_
#define _KernelMatrices_H_

#include <SharkDefs.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/Model.h>
#include <ReClaM/KernelFunction.h>
#include <Array/Array2D.h>
#include <vector>
#include <cmath>


////////////////////////////////////////////////////////////////////////////////


//! \brief Abstract base class of all quadratic program solvers
class QPSolver
{
public:
	QPSolver();
	virtual ~QPSolver();
};


//! \brief encapsulation of a quadratic matrix for quadratic programming
//!
//! \par
//! A #QPMatrix has two properties:
//! It provides access to its single entries and it
//! is able to swap any of its rows and columns.
//! In particular, this interface does not require
//! the matrix to fit into working memory.
class QPMatrix
{
public:
	//! Constructor
	QPMatrix(unsigned int size);

	//! Destructor
	virtual ~QPMatrix();

	//! \brief Return a single matrix element
	//!
	//! \par
	//! This method returns a single matrix element.
	//!
	//! \param  i  matrix row
	//! \param  j  matrix column
	//!
	virtual float Entry(unsigned int i, unsigned int j) = 0;

	//! \brief Exchange the rows i and j and the columns i and j
	//!
	//! \par
	//! W.l.o.g. it is assumed that \f$ i \leq j \f$.
	//! It may be advantageous for caching to reorganize
	//! the column order. In order to keep symmetric matrices
	//! symmetric the rows are swapped, too.
	//!
	//! \param  i  first row/column index
	//! \param  j  second row/column index
	//!
	virtual void FlipColumnsAndRows(unsigned int i, unsigned int j) = 0;

	//! return the size of the quadratic matrix
	inline unsigned int getMatrixSize() const
	{
		return matrixsize;
	}
		
	//! get number of times #Entry was called
	inline SharkInt64 getAccessCount() const
	{
		return accessCount;
	}
	
	//! set #accessCount to zero
	inline void resetAccessCount()
	{
		accessCount = 0;
	}
	
protected:
	//! length of each edge of the quadratic matrix
	unsigned int matrixsize;
	
	//! entry access counter
	SharkInt64 accessCount;
	
};


//! \brief Kernel Gram matrix
//!
//! \par
//! The #KernelMatrix is the most prominent subclass of
//! #QPMatrix providing the Gram matrix of a fixed data
//! set with respect to a kernel function inner product.
class KernelMatrix : public QPMatrix
{
public:
	//! Constructor
	//! \param kernelfunction   kernel function defining the Gram matrix
	//! \param data			 data to evaluate the kernel function
	//! \param count			should the number of kernel evaluations be counted?
	KernelMatrix(KernelFunction* kernelfunction,
				 const Array<double>& data,
				 bool count = false );

	//! Destructor
	~KernelMatrix();

	//! overriden virtual function, see #QPMatrix
	float Entry(unsigned int i, unsigned int j);

	//! overriden virtual function, see #QPMatrix
	void FlipColumnsAndRows(unsigned int i, unsigned int j);
	
protected:
	//! Kernel function defining the kernel Gram matrix
	KernelFunction* kernel;
	
	//! Array of data vectors for kernel evaluations
	Array<ArrayReference<double>* > x;
	
	//! should the number of accesses to #Entry be counted?
	bool countAccess;

};


//! \brief Precomputed Kernel Gram matrix
//!
//! \par
//! The #PrecomputedKernelMatrix class may be beneficial
//! for certain model selection strategies, in particular
//! if the kernel is fixed and the regularization parameter
//! is varied.
//!
//! \par
//! This class is a memory intensive alternative to the
//! plain KernelMatrix. It should be used only for problem
//! sizes of up to a few thousand examples. This class is
//! not yet well integrated with the CachedMatrix class,
//! such that data is redundantly.
//!
//! \par
//! The SVM_Optimizer class does not support this option
//! yet.
class PrecomputedKernelMatrix : public KernelMatrix
{
public:
	//! Constructor
	//! \param kernelfunction   kernel function defining the Gram matrix
	//! \param data			 data to evaluate the kernel function
	PrecomputedKernelMatrix(KernelFunction* kernelfunction,
				 const Array<double>& data);

	//! Destructor
	~PrecomputedKernelMatrix();

	//! overriden virtual function, see #QPMatrix
	float Entry(unsigned int i, unsigned int j);

	//! overriden virtual function, see #QPMatrix
	void FlipColumnsAndRows(unsigned int i, unsigned int j);

protected:
	Array<float> matrix;
	std::vector<unsigned int> permutation;
};


//! \brief Kernel Gram matrix
//!
//! \par
//! Regularized version of #KernelMatrix. The regularization
//! is achieved adding a vector to the matrix diagonal.
//! In particular, this is useful for support vector machines
//! with 2-norm penalty term.
class RegularizedKernelMatrix : public KernelMatrix
{
public:
	//! Constructor
	//! \param kernel		  kernel function
	//! \param data			 data to evaluate the kernel function
	//! \param diagModification vector d of diagonal modifiers
	RegularizedKernelMatrix(KernelFunction* kernel,
							const Array<double>& data,
							const Array<double>& diagModification);

	//! Destructor
	~RegularizedKernelMatrix();

	//! overriden virtual function, see #QPMatrix
	float Entry(unsigned int i, unsigned int j);

	//! overriden virtual function, see #QPMatrix
	void FlipColumnsAndRows(unsigned int i, unsigned int j);

protected:
	//! modification of the diagonal entries
	Array<double> diagMod;
};


//! \brief SVM regression matrix
//!
//! \par
//! The QPMatrix2 class is a \f$ 2n \times 2n \f$
//! block matrix of the form<br>
//! &nbsp;&nbsp;&nbsp; \f$ \left( \begin{array}{lr} M & M \\ M & M \end{array} \right) \f$ <br>
//! where M is an \f$ n \times n \f$ #QPMatrix.
//! This matrix form is needed in support vector machine
//! regression problems.
class QPMatrix2 : public QPMatrix
{
public:
	//! Constructor.
	//! \param base  underlying matrix M, see class description #QPMatrix2.
	//!
	//! \par
	//! The destructor will destroy the base matrix.
	QPMatrix2(QPMatrix* base);

	//! Destructor
	~QPMatrix2();

	//! overriden virtual function, see #QPMatrix
	float Entry(unsigned int i, unsigned int j);

	//! overriden virtual function, see #QPMatrix
	void FlipColumnsAndRows(unsigned int i, unsigned int j);

protected:
	//! underlying #KernelMatrix object
	QPMatrix* baseMatrix;

	//! coordinate permutation
	Array<unsigned int> mapping;
};


//! \brief Efficient quadratic matrix cache
//!
//! \par
//! The access operations of the CachedMatrix class
//! are specially tuned towards the iterative solution
//! of quadratic programs resulting in sparse solutions.
//!
//! \par
//! In contrast to the #QPMatrix base class, the
//! desired access operation is #Row.
//! Algorithms should try to rely on the #Row method for
//! retreiving matrix values because the #Entry method
//! does not cache its computations.
class CachedMatrix : public QPMatrix
{
public:
	//! Constructor
	//! \param base	   Matrix to cache
	//! \param cachesize  Main memory to use as a kernel cache, in floats. Default is 256MB
	CachedMatrix(QPMatrix* base, unsigned int cachesize = 0x4000000);

	//! Destructor
	~CachedMatrix();

	//! \brief Return a subset of a matrix row
	//!
	//! \par
	//! This method returns an array of float with at least
	//! the entries in the interval [begin, end[ filled in.
	//! If #temp is set to true, the computed values are not
	//! stored in the cache.
	//!
	//! \param k	  matrix row
	//! \param begin  first column to be filled in
	//! \param end	last column to be filled in +1
	//! \param temp   are the return values temporary or should they be cached?
	//! \param keep   if there are more entries stored than requested, and cache is not full, keep them?
	//   keep is useful in combination with shrinking: if some variables were disabled but maybe needed,
	//   keep=false would unnecessarily throw their cache entries away.
	float* Row(unsigned int k, unsigned int begin, unsigned int end, bool temp = false, bool keep = false);

	//! overriden virtual function
	float Entry(unsigned int i, unsigned int j);

	//! \brief Exchange the rows i and j and the columns i and j
	//!
	//! \par
	//! W.l.o.g. it is assumed that \f$ i \leq j \f$.
	//! It may be advantageous for caching to reorganize
	//! the column order. In order to keep symmetric matrices
	//! symmetric the rows are swapped, too.
	//!
	//! \param  i  first row/column index
	//! \param  j  second row/column index
	//!
	void FlipColumnsAndRows(unsigned int i, unsigned int j);

	inline unsigned int getMaxCacheSize() const
	{
		return cacheMaxSize;
	}

	inline unsigned int getCacheSize() const
	{
		return cacheSize;
	}

	inline unsigned int getCacheRowSize(unsigned int k)
	{
		return cacheEntry[k].length;
	}

	inline void Clear()
	{ cacheClear(); }

	void CacheRowResize(unsigned int k, unsigned int newsize);
	void CacheRowRelease(unsigned int k);
	
	//! get number of kernel evaluations
	inline SharkInt64 getBaseMatrixKernelEvals()
	{
		return baseMatrix->getAccessCount();
	}
	
	//! set kernel evaluation counter to zero
	inline void resetBaseMatrixKernelEvals()
	{
		baseMatrix->resetAccessCount();
	}
	
protected:
	//! matrix to be cached
	QPMatrix* baseMatrix;

	// matrix cache
	unsigned int cacheSize;						// current cache size in floats
	unsigned int cacheMaxSize;					// maximum cache size in floats
	struct tCacheEntry							// cache data held for every example
	{
		float* data;							// float array containing a matrix row
		int length;								// length of this matrix row
		int older;								// next older entry. -2 for uninitialized, -1 for "this is oldest"
		int newer;								// next newer entry. -2 for uninitialized, -1 for "this is newest"
	};
	std::vector<tCacheEntry> cacheEntry;		// cache entry description
	std::vector<float> cacheTemp;				// single kernel row
	int cacheNewest;							// index of the newest entry
	int cacheOldest;							// index of the oldest entry
	
	//! append the entry to the ordered list
	void cacheAppend(int var);

	//! remove the entry from the ordered list
	void cacheRemove(int var);

	//! add an entry to the cache and append
	//! it to the ordered list
	void cacheAdd(int var, unsigned int length);

	//! remove an entry from the cache and the ordered list
	void cacheDelete(int var);

	//! resize a cache entry
	void cacheResize(int var, unsigned int newlength);

	//! completely clear the cache
	void cacheClear();
};

#endif
