//===========================================================================
/*!
 *  \file QuadraticProgram.h
 *
 *  \brief Quadratic programming for Support Vector Machines
 *
 *
 *  \par
 *  This file provides the following interfaces:
 *  <ul>
 *    <li>A quadratic matrix interface</li>
 *    <li>Several special matrices based on kernels and other matrices</li>
 *    <li>A quadtaric matrix cache</li>
 *    <li>A quadratic program solver for a special #SVM related family of problems</li>
 *  </ul>
 *  All methods are specifically tuned toward the solution of
 *  quadratic programs occuring in support vector machines and
 *  related kernel methods. There is no support the general
 *  quadratic programs. Refer to the #QuadraticProgram class
 *  documentation for details.
 *
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


#ifndef _QuadraticProgram_H_
#define _QuadraticProgram_H_


#include <SharkDefs.h>
#include <ReClaM/Model.h>
#include <ReClaM/KernelFunction.h>
#include <Array/Array2D.h>
#include <vector>
#include <cmath>
#include<time.h>


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

protected:
	//! length of each edge of the quadratic matrix
	unsigned int matrixsize;
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
	//! \param data             data to evaluate the kernel function
	KernelMatrix(KernelFunction* kernelfunction,
				 const Array<double>& data);

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
	//! \param kernel          kernel function
	//! \param data             data to evaluate the kernel function
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
	//! \param base       Matrix to cache
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
	//! \param k      matrix row
	//! \param begin  first column to be filled in
	//! \param end    last column to be filled in +1
	//! \param temp   are the return values temporary or should they be cached?
	float* Row(unsigned int k, unsigned int begin, unsigned int end, bool temp = false);

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
		int older;								// next older entry
		int newer;								// next newer entry
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


//!
//! \brief Quadratic program solver for binary SVMs
//!
//! \par
//! The QpSvmDecomp class is a decomposition-based solver
//! for the quadratic program occuring when training a
//! standard binary support vector machine (SVM).
//! This problem has the following structure (for
//! \f$ \alpha \in R^{\ell} \f$):
//!
//! \par
//! maximize \f$ W(\alpha) = v^T \alpha - \frac{1}{2} \alpha^T M \alpha \f$<br>
//! s.t. \f$ \sum_{i=1}^{\ell} \alpha_i = z \f$ (equality constraint)<br>
//! and \f$ l_i \leq \alpha_i \leq u_i \f$ for all \f$ 1 \leq i \leq \ell \f$ (box constraints).
//!
//! \par
//! Here, z is a number, v is any vector and, M is a
//! positive definite symmetric quadratic matrix.
//! \f$ l_i \leq u_i \f$ are lower and upper bounds
//! on the variables.
//!
//! \par
//! The quadratic program is special in that it has a
//! special box form of its inequality constraints, and
//! that is has a single specially aligned equality
//! constraint. These properties make the SMO algorithm
//! (Platt, 1999) and variants thereof (Fan et al., 2005;
//! Glasmachers and Igel, 2006) directly applicable.
//!
//! \par
//! This solver uses the basic SMO algorithm with
//! caching and shrinking techniques (Joachims, 1998) and
//! a switching between two highly efficient worling set
//! selection algorithms based on second order information.
//! Because the SelectWorkingSet method is virtual,
//! it is easy to implement other strategies if needed.
//!
//! \par
//! For practical considerations the solver supports
//! several stopping conditions. Usually, the optimization
//! stops if the Karush-Kuhn-Tucker (KKT) condition for
//! optimality are satisfied up to a certain accuracy.
//! In the case the optimal function value is known a
//! priori it is possible to stop as soon as the objective
//! is above a given threshold. In both cases it is very
//! difficult to predict the runtime. Therefore the
//! solver can stop after a predefined number of
//! iterations or after a predefined time period. In
//! these cases the solution found will not be near
//! optimal. Usually this happens only during model
//! selection while investigating hyperparameters with
//! poor generatlization ability.
//!
class QpSvmDecomp : public QPSolver
{
public:
	//! Constructor
	//! \param  quadratic  quadratic part of the objective function and matrix cache
	QpSvmDecomp(CachedMatrix& quadraticPart);

	//! Destructor
	virtual ~QpSvmDecomp();

	//!
	//! \brief solve the quadratic program
	//!
	//! \par
	//! This is the core method of the #QpSvmDecomp class.
	//! It computes the solution of the problem defined by the
	//! parameters. This interface allows for the solution of
	//! multiple problems with the same quadratic part, reusing
	//! the matrix cache, but with arbirtrary linear part, box
	//! constraints and stopping conditions.
	//!
	//! \param linearPart       linear part v of the target function
	//! \param boxLower         vector l of lower bounds
	//! \param boxUpper         vector u of upper bounds
	//! \param solutionVector   input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param eps              solution accuracy, in terms of the maximum KKT violation
	//! \param threshold        threshold to use for the objective value stopping criterion
	//! \return                 objective value at the optimum
	//!
	double Solve(const Array<double>& linearPart,
				 const Array<double>& boxLower,
				 const Array<double>& boxUpper,
				 Array<double>& solutionVector,
				 double eps = 0.001,
				 double threshold = 1e100);

	//!
	//! \brief compute the inner product of a training example with a linear combination of the training examples
	//!
	//! \par
	//! This method computes the inner product of
	//! a training example with a linear combination
	//! of all training examples. This computation
	//! is fast as it makes use of the kernel cache
	//! if possible.
	//!
	//! \param  index  index of the training example
	//! \param  coeff  list of coefficients of the training examples
	//! \return        result of the inner product
	//!
	double ComputeInnerProduct(unsigned int index, const Array<double>& coeff);

	//!
	//! \brief return the gradient of the objective function in the optimum
	//!
	//! \param  grad  gradient of the objective function
	void getGradient(Array<double>& grad);

	//! enable/disable console status output
	inline void setVerbose(bool verbose = false)
	{
		printInfo = verbose;
	}

	//! set the working set strategy parameter
	inline void setStrategy(const char* strategy = NULL)
	{
		WSS_Strategy = strategy;
	}

	//! enable/disable shrinking
	inline void setShrinking(bool shrinking = true)
	{
		useShrinking = shrinking;
	}

	//! Return the number of iterations used to solve the problem
	inline SharkInt64 iterations()
	{
		return iter;
	}

	//! Is the solution optimal up to the given epsilon?
	inline bool isOptimal()
	{
		return optimal;
	}

	//! set the maximal number of iterations
	//! the solver is allowed to perform
	//! \param  maxiter  maximal number of iterations, -1 for infinity
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	//! set the maximal number of seconds
	//! the solver is allowed to run
	//! \param  seconds  maximal number of seconds, -1 for infinity
	inline void setMaxSeconds(int seconds = -1)
	{
		maxSeconds = seconds;
	}

protected:
	//! problem dimension
	unsigned int dimension;

	//! representation of the quadratic part of the objective function
	CachedMatrix& quadratic;

	//! linear part of the objective function
	Array<double> linear;

	//! box constraint lower bound, that is, minimal variable value
	Array<double> boxMin;

	//! box constraint upper bound, that is, maximal variable value
	Array<double> boxMax;

	//! solution statistics: number of iterations
	SharkInt64 iter;

	//! is the solution found (near) optimal
	//! or has the solver stopped due to an
	//! interation or time constraint?
	bool optimal;

	//! maximum number of iterations
	SharkInt64 maxIter;

	//! maximum number of seconds
	int maxSeconds;

	//! should the solver print its status to the standard output?
	bool printInfo;

	//! working set selection strategy to follow
	const char* WSS_Strategy;

	//! should the solver use the shrinking heuristics?
	bool useShrinking;

	//! pointer to the currently used working set selection algorithm
	bool(QpSvmDecomp::*currentWSS)(unsigned int&, unsigned int&);

	//! number of currently active variables
	unsigned int active;

	//! permutation of the variables alpha, gradient, etc.
	Array<unsigned int> permutation;

	//! diagonal matrix entries
	//! The diagonal array is of fixed size and not subject to shrinking.
	Array<double> diagonal;

	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	Array<double> gradient;

	//! Solution candidate
	Array<double> alpha;

	//! indicator of the first decomposition iteration
	bool bFirst;

	//! first component of the previous working set
	unsigned int old_i;

	//! second component of the previous working set
	unsigned int old_j;

	//! stopping condition - solution accuracy
	double epsilon;

	//! \brief Select the most violatig pair (MVP)
	//!
	//! \return true if the solution is already sufficiently optimal
	//!  \param i  first working set component
	//!  \param j  second working set component
	bool MVP(unsigned int& i, unsigned int& j);

	//! \brief Select a working set according to the hybrid maximum gain (HMG) algorithm
	//!
	//! \return true if the solution is already sufficiently optimal
	//!  \param i  first working set component
	//!  \param j  second working set component
	bool HMG(unsigned int& i, unsigned int& j);

	//! \brief Select a working set according to the second order algorithm of LIBSVM 2.8
	//!
	//! \return true if the solution is already sufficiently optimal
	//!  \param i  first working set component
	//!  \param j  second working set component
	bool Libsvm28(unsigned int& i, unsigned int& j);

	//! \brief Select a working set
	//!
	//! \par
	//! This member is implemented as a wrapper to HMG.
	//! \return true if the solution is already sufficiently optimal
	//!  \param i  first working set component
	//!  \param j  second working set component
	virtual bool SelectWorkingSet(unsigned int& i, unsigned int& j);

	//! Choose a suitable working set algorithm
	void SelectWSS();

	//! Shrink the problem
	void Shrink();

	//! Active all variables
	void Unshrink(bool complete = false);

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

	//! exchange two variables via the permutation
	void FlipCoordinates(unsigned int i, unsigned int j);
};


//!
//! \brief Quadratic program solver for box constrained problems
//!
//! \par
//! This algorithm solves the same problem as QpSvmDecomp but
//! without the equality constraint. For example, this problem
//! corresponds to SVMs without bias. Similar techniques are
//! applied, but this solver solves more complicated sub-problems.
//!
class QpBoxDecomp : public QPSolver
{
public:
	//! Constructor
	//! \param  quadratic  quadratic part of the objective function and matrix cache
	QpBoxDecomp(CachedMatrix& quadraticPart);

	//! Destructor
	virtual ~QpBoxDecomp();

	//!
	//! \brief solve the quadratic program
	//!
	//! \param linearPart       linear part v of the target function
	//! \param boxLower         vector l of lower bounds
	//! \param boxUpper         vector u of upper bounds
	//! \param solutionVector   input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param eps              solution accuracy, in terms of the maximum KKT violation
	//!
	void Solve(const Array<double>& linearPart,
				 const Array<double>& boxLower,
				 const Array<double>& boxUpper,
				 Array<double>& solutionVector,
				 double eps = 0.001);

	//! Return the number of iterations used to solve the problem
	inline SharkInt64 iterations()
	{
		return iter;
	}

	//! Is the solution optimal up to the given epsilon?
	inline bool isOptimal()
	{
		return optimal;
	}

	//! set the maximal number of iterations
	//! the solver is allowed to perform
	//! \param  maxiter  maximal number of iterations, -1 for infinity
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	inline void Set_WSS_Strategy(int i = 1)
	{
		RANGE_CHECK(i == 1 || i == 2);
		WSS_Strategy = i;
	}

protected:
	//! Internally used by Solve2D;
	//! computes the solution of a
	//! one-dimensional sub-problem.
	double StepEdge(double alpha, double g, double Q, double L, double U, double& mu);

	//! Exact solver for a two-dimensional sub-problem.
	void Solve2D(double alphai, double alphaj,
					double gi, double gj,
					double Qii, double Qij, double Qjj,
					double Li, double Ui, double Lj, double Uj,
					double& mui, double& muj);

	//! decomposition loop
	void Loop();

	//! problem dimension
	unsigned int dimension;

	//! representation of the quadratic part of the objective function
	CachedMatrix& quadratic;

	//! linear part of the objective function
	Array<double> linear;

	//! box constraint lower bound, that is, minimal variable value
	Array<double> boxMin;

	//! box constraint upper bound, that is, maximal variable value
	Array<double> boxMax;

	//! number of currently active variables
	unsigned int active;

	//! permutation of the variables alpha, gradient, etc.
	Array<unsigned int> permutation;

	//! diagonal matrix entries
	//! The diagonal array is of fixed size and not subject to shrinking.
	Array<double> diagonal;

	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	Array<double> gradient;

	//! Solution candidate
	Array<double> alpha;

	//! stopping condition - solution accuracy
	double epsilon;

	//! number of variables in the working set (1 or 2)
	int WSS_Strategy;

	//! working set selection in the case of a single variable
	virtual bool SelectWorkingSet(unsigned int& i);

	//! working set selection in the case of two variables
	virtual bool SelectWorkingSet(unsigned int& i, unsigned int& j);

	//! Shrink the problem
	void Shrink();

	//! Active all variables
	void Unshrink(bool complete = false);

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

	//! solution statistics: number of iterations
	SharkInt64 iter;

	//! is the solution found (near) optimal
	//! or has the solver stopped due to an
	//! interation or time constraint?
	bool optimal;

	//! maximum number of iterations
	SharkInt64 maxIter;

	//! exchange two variables via the permutation
	void FlipCoordinates(unsigned int i, unsigned int j);
};


//!
//! \brief Quadratic program solver for box constrained multi class problems
//!
//! \par
//! This algorithm solves the same problem as QpBoxDecomp,
//! with the difference that the structure of the quadratic
//! program of multi class classification is exploited for
//! more efficient caching and a more natural problem
//! encoding.
//!
class QpMcDecomp : public QPSolver
{
public:
	//! Constructor
	//! \param  kernel  kernel matrix cache
	QpMcDecomp(CachedMatrix& kernel);

	//! Destructor
	~QpMcDecomp();

	//!
	//! \brief solve the quadratic program
	//!
	//! \param  classes         number of classes
	//! \param  modifiers       list of 64 kernel modifiers, see method QpMcDecomp::Modifier
	//! \param  target          class label indices in {0, ..., classes-1}
	//! \param  linearPart      linear part of the objective function
	//! \param  lower           coordinate-wise lower bound
	//! \param  upper           coordinate-wise upper bound
	//! \param  solutionVector  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  eps             solution accuracy, in terms of the maximum KKT violation
	//!
	void Solve(unsigned int classes,
					const double* modifiers,
					const Array<double>& target,
					const Array<double>& linearPart,
					const Array<double>& lower,
					const Array<double>& upper,
					Array<double>& solutionVector,
					double eps);

	//! Return the number of iterations used to solve the problem
	inline SharkInt64 iterations()
	{
		return iter;
	}

	//! Is the solution optimal up to the given epsilon?
	inline bool isOptimal()
	{
		return optimal;
	}

	//! set the maximal number of iterations
	//! the solver is allowed to perform
	//! \param  maxiter  maximal number of iterations, -1 for infinity
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	//! Set number of variables in the working set,
	//! accepted values are one or two.
	inline void Set_WSS_Strategy (int i = 1)
	{
		RANGE_CHECK(i == 1 || i == 2);
		WSS_Strategy = i;
	}

protected:
	//! Return the "kernel modifier" which is used to
	//! compute the Q-matrix form the kernel matrix.
	inline double Modifier(unsigned int yi, unsigned int yj, unsigned int vi, unsigned int vj) const
	{
		unsigned int index = 0;
		if (yi == yj) index += 1;
		if (yi == vi) index += 2;
		if (yi == vj) index += 4;
		if (yj == vi) index += 8;
		if (yj == vj) index += 16;
		if (vi == vj) index += 32;
		return m_modifier[index];
	}

	//! Internally used by Solve2D;
	//! computes the solution of a
	//! one-dimensional sub-problem.
	double StepEdge(double alpha, double g, double Q, double L, double U, double& mu);

	//! Exact solver for a two-dimensional sub-problem.
	void Solve2D(double alphai, double alphaj,
					double gi, double gj,
					double Qii, double Qij, double Qjj,
					double Li, double Ui, double Lj, double Uj,
					double& mui, double& muj);

	//! data structure describing one training example
	struct tExample
	{
		unsigned int index;			// example index
		unsigned int label;			// label of this example
		unsigned int active;		// number of active variables
		unsigned int* variables;	// list of variables, active ones first
	};

	//! data structure describing one variable of the problem
	struct tVariable
	{
		unsigned int example;		// index into the example list
		unsigned int index;			// index into example->variables
		unsigned int label;			// label corresponding to this variable
		double diagonal;			// diagonal entry of the big Q-matrix
	};

	//! information about each training example
	std::vector<tExample> example;

	//! information about each variable of the problem
	std::vector<tVariable> variable;

	//! space for the example[i].variables pointers
	std::vector<unsigned int> storage;

	//! number of examples in the problem (size of the kernel matrix)
	unsigned int examples;

	//! number of classes in the problem
	unsigned int classes;

	//! number of variables in the problem = examples times classes
	unsigned int variables;

	//! kernel matrix cache
	CachedMatrix& kernelMatrix;

	//! linear part of the objective function
	Array<double> linear;

	//! box constraint lower bound, that is, minimal variable value
	Array<double> boxMin;

	//! box constraint upper bound, that is, maximal variable value
	Array<double> boxMax;

	//! number of currently active examples
	unsigned int activeEx;

	//! number of currently active variables
	unsigned int activeVar;

	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	Array<double> gradient;

	//! Solution candidate
	Array<double> alpha;

	//! stopping condition - solution accuracy
	double epsilon;

	//! select an optimization direction
	virtual bool SelectWorkingSet(unsigned int& i);

	//! select  optimization directions i and j
	virtual bool SelectWorkingSet(unsigned int& i, unsigned int& j);

	//! Shrink the problem
	void Shrink();

	//! Active all variables
	void Unshrink(bool complete = false);

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

	//! shrink a variable
	void DeactivateVariable(unsigned int v);

	//! shrink an examples
	void DeactivateExample(unsigned int e);

	//! solution statistics: number of iterations
	SharkInt64 iter;

	//! is the solution found (near) optimal
	//! or has the solver stopped due to an
	//! interation or time constraint?
	bool optimal;

	//! maximum number of iterations
	SharkInt64 maxIter;

	//! number of variables in the working set
	int WSS_Strategy;

	//! kernel modifiers under six binary conditions
	double m_modifier[64];
};


//!
//! \brief Quadratic program solver for the multi class
//!        SVM proposed by Crammer and Singer
//!
//! \par
//! This algorithm solves the same quadratic program introduced in:
//!
//! K. Crammer, Y. Singer.
//! <a href="http://www.jmlr.org/papers/volume2/crammer01a/crammer01a.pdf">
//! On the Algorithmic Implementation of Multiclass Kernel-based
//! Vector Machines.</a>
//! In Journal of Machine Learning Research, pp. 265-292 (2001).
//!
//! This problem is special among mutli-class SVMs in that it has
//! as many equality constraints as there are training examples.
//! It is solved with a straight-forward variant of the SMO scheme
//! (Platt, 1999).
//!
class QpCrammerSingerDecomp : public QPSolver
{
public:
	//! Constructor
	//! \param  kernel   kernel matrix cache
	//! \param  y        classification targets
	//! \param  classes  number of classes
	QpCrammerSingerDecomp(CachedMatrix& kernel, const Array<double>& y, unsigned int classes);

	//! Destructor
	virtual ~QpCrammerSingerDecomp();

	//!
	//! \brief solve the quadratic program
	//!
	//! \param  solutionVector  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  beta            regularization constant (corresponding to 1/C in the C-SVM)
	//! \param  eps             solution accuracy, in terms of the maximum KKT violation
	//!
	void Solve(Array<double>& solutionVector,
				double beta,
				double eps = 0.001);

	//! Return the number of iterations used to solve the problem
	inline SharkInt64 iterations()
	{
		return iter;
	}

	//! Is the solution optimal up to the given epsilon?
	inline bool isOptimal()
	{
		return optimal;
	}

	//! set the maximal number of iterations
	//! the solver is allowed to perform
	//! \param  maxiter  maximal number of iterations, -1 for infinity
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

protected:
	//! data structure describing one training example
	struct tExample
	{
		unsigned int index;			// example index
		unsigned int label;			// label of this example
		unsigned int active;		// number of active variables
		unsigned int* variables;	// list of variables, active ones first
	};

	//! data structure describing one variable of the problem
	struct tVariable
	{
		unsigned int example;		// index into the example list
		unsigned int index;			// index into example->variables
		unsigned int label;			// label corresponding to this variable
	};

	//! information about each training example
	std::vector<tExample> example;

	//! information about each variable of the problem
	std::vector<tVariable> variable;

	//! space for the example[i].variables pointers
	std::vector<unsigned int> storage;

	unsigned int examples;
	unsigned int classes;
	unsigned int variables;

	//! kernel matrix cache
	CachedMatrix& kernelMatrix;

	//! linear part of the objective function
	Array<double> linear;

	//! box constraint lower bound, that is, minimal variable value
	Array<double> boxMin;

	//! box constraint upper bound, that is, maximal variable value
	Array<double> boxMax;

	//! number of currently active examples
	unsigned int activeEx;

	//! number of currently active variables
	unsigned int activeVar;

	//! diagonal matrix entries
	//! The diagonal array is of fixed size and not subject to shrinking.
	Array<double> diagonal;

	//! gradient of the objective function
	//! The gradient array is of fixed size and not subject to shrinking.
	Array<double> gradient;

	//! Solution candidate
	Array<double> alpha;

	//! stopping condition - solution accuracy
	double epsilon;

	double SelectWorkingSet(unsigned int& i, unsigned int& j);

	//! Shrink the problem
	void Shrink();

	//! Active all variables
	void Unshrink(bool complete = false);

	//! true if the problem has already been unshrinked
	bool bUnshrinked;

      	//! true if the the user wants to use shrinking mechanism (default = true)
	bool shrinkCheck;

	//! shrink a variable
	void DeactivateVariable(unsigned int v);

	//! shrink an examples
	void DeactivateExample(unsigned int e);

	//! solution statistics: number of iterations
	SharkInt64 iter;

	//! is the solution found (near) optimal
	//! or has the solver stopped due to an
	//! interation or time constraint?
	bool optimal;

	//! maximum number of iterations
	SharkInt64 maxIter;
};


#endif
