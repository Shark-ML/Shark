//===========================================================================
/*!
 *  \file QpSvmDecomp.h
 *
 *  \brief Quadratic programming for binary Support Vector Machines
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


#ifndef _QpSvmDecomp_H_
#define _QpSvmDecomp_H_

class CachedMatrix;
class QPSolver;

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
	//! \param linearPart	   linear part v of the target function
	//! \param boxLower		 vector l of lower bounds
	//! \param boxUpper		 vector u of upper bounds
	//! \param solutionVector   input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param eps			  solution accuracy, in terms of the maximum KKT violation
	//! \param threshold		threshold to use for the objective value stopping criterion
	//! \return				 objective value at the optimum
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
	//! \return		result of the inner product
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

#endif
