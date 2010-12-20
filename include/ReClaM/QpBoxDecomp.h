//===========================================================================
/*!
 *  \file QpBoxDecomp.h
 *
 *  \brief Quadratic programming for binary Support Vector Machines with box constraints
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


#ifndef _QpBoxDecomp_H_
#define _QpBoxDecomp_H_

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
	//! \param linearPart	   linear part v of the target function
	//! \param boxLower		 vector l of lower bounds
	//! \param boxUpper		 vector u of upper bounds
	//! \param solutionVector   input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param eps			  solution accuracy, in terms of the maximum KKT violation
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

#endif
