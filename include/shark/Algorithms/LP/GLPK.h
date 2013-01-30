//===========================================================================
/*!
 *  \file GLPK.h
 *
 *  \brief Shark adaptation of the GNU Linear Programming Kit
 *
 *
 *  \par
 *  This file contains a minimal adaptation of GLPK, the GNU
 *  Linear Programming Kit, version 4.45. The functionality has
 *  been reduced to the simplex solver.
 *  The original copyright notice is found below:
 *
 *  \par
 *  \par
 *  This code is part of GLPK (GNU Linear Programming Kit).<br>
 *  <br>
 *  Copyright (C) 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,<br>
 *  2009, 2010 Andrew Makhorin, Department for Applied Informatics,<br>
 *  Moscow Aviation Institute, Moscow, Russia. All rights reserved.<br>
 *  E-mail: &lt;mao@gnu.org&gt;.<br>
 *  <br>
 *  GLPK is free software: you can redistribute it and/or modify it<br>
 *  under the terms of the GNU General Public License as published by<br>
 *  the Free Software Foundation, either version 3 of the License, or<br>
 *  (at your option) any later version.<br>
 *  <br>
 *  GLPK is distributed in the hope that it will be useful, but WITHOUT<br>
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY<br>
 *  or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public<br>
 *  License for more details.<br>
 *  <br>
 *  You should have received a copy of the GNU General Public License<br>
 *  along with GLPK. If not, see &lt;http://www.gnu.org/licenses/&gt;.<br>
 */
//===========================================================================


#ifndef SHARK_ML_LP_GLPK
#define SHARK_ML_LP_GLPK


#include <vector>


namespace shark {


namespace glpk
{
	struct glp_prob;
}


///
/// \brief Linear Program Solver
///
/// \par
/// The LP class is a minimalistic abstraction layer on
/// top of the GLPK simplex solver. It is designed to
/// solve linear programs, that is, problems of the type
///
/// \par
/// \f$ \min/\max v^T x \f$ <br>
/// \f$ s.t. l \leq   x \leq u \f$ <br>
/// \f$ s.t. L \leq A x \leq U \f$
///
/// \par
/// All inequalities are component-wise, and lower and
/// upper bounds may be \f$ \pm \infty \f$.
///
/// \par
/// We refer to the GLPK library, version 4.45, for more
/// detailed documentation.
///
/// \par
/// <b>IMPORTANT:</b><br>
/// In contrast to common practice in c/c++ programming,
/// indices in GLPK are one-based, as opposed to
/// zero-based! Thus, 1 has to be added to all row and
/// column indices, and all arrays have to be one
/// element longer, with the 0-th element unused.
///
class LP
{
public:
	/// initialize the linear program object
	LP();

	/// destroy the linear program object
	~LP();

	/// set the optimization direction to minimization (default)
	void setMinimize();

	/// set the optimization direction to maximization
	void setMaximize();

	/// add a number of rows (variables used to express constraints) to the problem
	void addRows(unsigned int rows);

	/// add a number of columns (variables) to the problem
	void addColumns(unsigned int cols);

	/// set the coefficient of the objective function
	/// (\f$ v_n \f$ in the above problem statement)
	void setObjectiveCoefficient(unsigned int col, double coeff);

	/// declare that the row variable should be unbounded
	/// (this rarely makes sense...)
	void setRowFree(unsigned int row);

	/// declare that the row variable should be lower bounded
	void setRowLowerBounded(unsigned int row, double lower);

	/// declare that the row variable should be upper bounded
	void setRowUpperBounded(unsigned int row, double upper);

	/// declare that the row variable should be bounded from
	/// below and above
	void setRowDoubleBounded(unsigned int row, double lower, double upper);

	/// declare that the row variable should be fixed to a
	/// single value
	void setRowFixed(unsigned int row, double value);

	/// declare that the column variable should be unbounded
	void setColumnFree(unsigned int col);

	/// declare that the column variable should be lower bounded
	void setColumnLowerBounded(unsigned int col, double lower);

	/// declare that the column variable should be upper bounded
	void setColumnUpperBounded(unsigned int col, double upper);

	/// declare that the column variable should be bounded from
	/// below and above
	void setColumnDoubleBounded(unsigned int col, double lower, double upper);

	/// declare that the column variable should be fixed to a
	/// single value (this rarely makes sense...)
	void setColumnFixed(unsigned int col, double value);

	/// Define the matrix A connecting row and column variables.
	/// The matrix is represented in a sparse format, with each
	/// entry in the arrays row, columns, and value representing
	/// a non-zero entry of the matrix as
	/// A(row[n], column[n]) = value[n]
	void setConstraintMatrix(std::vector<unsigned int> const& row, std::vector<unsigned int> const& col, std::vector<double> const& value);

	/// Set the status of the row variable as either basic or not.
	/// There must be exactly as many independent basic variables
	/// as there are rows in the problem to define the initial
	/// solution.
	void setRowStatus(unsigned int row, bool basic);

	/// Set the status of the column variable as either basic or not.
	/// There must be exactly as many independent basic variables
	/// as there are rows in the problem to define the initial
	/// solution.
	void setColumnStatus(unsigned int col, bool basic);

	/// Solve the linear program with the simplex method.
	bool solve();

	/// return a coefficient of the solution
	double solution(unsigned int col);

protected:
	/// GLPK problem object
	glpk::glp_prob* m_prob;
};


} // namespace shark


#endif
