//===========================================================================
/*!
 *  \file QpMcDecomp.h
 *
 *  \brief Quadratic programming for Multi-Class Support Vector Machines
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


#ifndef _QpMcDecomp_H_
#define _QpMcDecomp_H_

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
	//! \param  classes		 number of classes
	//! \param  modifiers	   list of 64 kernel modifiers, see method QpMcDecomp::Modifier
	//! \param  target		  class label indices in {0, ..., classes-1}
	//! \param  linearPart	  linear part of the objective function
	//! \param  lower		   coordinate-wise lower bound
	//! \param  upper		   coordinate-wise upper bound
	//! \param  solutionVector  input: initial feasible vector \f$ \alpha \f$; output: solution \f$ \alpha^* \f$
	//! \param  eps			 solution accuracy, in terms of the maximum KKT violation
	//!
	void Solve(unsigned int classes,
					const double* modifiers,
					const Array<double>& target,
					const Array<double>& linearPart,
					const Array<double>& lower,
					const Array<double>& upper,
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

	//! Set the maximal number of iterations
	//! the solver is allowed to perform
	//! \param  maxiter  maximal number of iterations, -1 for infinity
	inline void setMaxIterations(SharkInt64 maxiter = -1)
	{
		this->maxIter = maxiter;
	}

	//! Set number of variables in the working set,
	//! acceptable values are one or two.
	inline void Set_WSS_Strategy(unsigned int i = 1)
	{
		RANGE_CHECK(i == 1 || i == 2);
		WSS_Strategy = i;
	}

protected:
	//! Return the 'kernel modifier' which is used to
	//! compute the Q-matrix from the kernel matrix.
	//! Let (i, m) and (j, n) denote the variables to
	//! compare, then the Q-matrix entry takes the form
	//! \f$ Q_{(i,m),(j,n)} = M(y_i, y_j, m, n) \cdot k(x_i, x_j) \f$,
	//! where M denotes the so-called modifier. In
	//! general, it depends on four label-valued terms.
	//! However, due to symmetry it depends only on
	//! the six possible binary equality relations
	//! among these, resulting in \f$ 2^6 = 64 \f$
	//! different values.
	//!
	//! \param  yi  label of the example corresponding to the first variable (i,m)
	//! \param  yj  label of the example corresponding to the second variable (j,n)
	//! \param  vi  label 'm' of the first variable
	//! \param  vj  label 'n' of the second variable
	inline double Modifier(unsigned int yi, unsigned int yj, unsigned int vi, unsigned int vj) const
	{
		unsigned int index = 0;
		if (yi == yj) index += 1;			// bit 0 indicates y_i = y_j
		if (vi == vj) index += 2;			// bit 1 indicates m = n
		if (yi == vj) index += 4;			// bit 2 indicates y_i = n
		if (yj == vi) index += 8;			// bit 3 indicates y_j = m
		if (yi == vi) index += 16;			// bit 4 indicates y_i = m
		if (yj == vj) index += 32;			// bit 5 indicates y_j = n
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

#endif
