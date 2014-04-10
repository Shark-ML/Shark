//===========================================================================
/*!
 * 
 *
 * \brief       Quadratic programming for Support Vector Machines
 * 
 * 
 * \par
 * This file provides a number of classes representing hugh dense
 * matrices all related to kernel Gram matices of possibly large
 * datasets. These classes share a common interface for
 * (a) providing a matrix entry,
 * (b) swapping two variable indices, and
 * (c) returning the matrix size.
 * 
 * \par
 * This interface is required by the template class CachedMatrix,
 * which provides a cache mechanism for restricted matrix rows, as it
 * is used by various quadratic program solvers within the library.
 * The PrecomputedMatrix provides a sometimes faster but more memory
 * intensive alternative to CachedMatrix.
 * 
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2007-2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================


#ifndef SHARK_ALGORITHMS_QP_QUADRATICPROGRAM_H
#define SHARK_ALGORITHMS_QP_QUADRATICPROGRAM_H

#include <cmath>


namespace shark {

/// reason for the quadratic programming solver
/// to stop the iterative optimization process
enum QpStopType
{
	QpNone = 0,
	QpAccuracyReached = 1,
	QpMaxIterationsReached = 4,
	QpTimeout = 8,
};


///
/// \brief stopping conditions for quadratic programming
///
/// \par
/// The QpStoppingCondition structure defines conditions
/// for stopping the optimization procedure.
///
//! \par
//! For practical considerations the solvers supports
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
//! optimal. In SVM training, using sensitive settings,
//! this should happen only during model selection while
//! investigating hyperparameters with poor
//! generalization ability.
//!
struct QpStoppingCondition
{
	/// Constructor
	QpStoppingCondition(double accuracy = 0.001, unsigned long long iterations = 0xffffffff, double value = 1e100, double seconds = 1e100)
	{
		minAccuracy = accuracy;
		maxIterations = iterations;
		targetValue = value;
		maxSeconds = seconds;
	}

	/// minimum accuracy to be achieved, usually KKT violation
	double minAccuracy;

	/// maximum number of decomposition iterations (default to 0 - not used)
	unsigned long long maxIterations;

	/// target objective function value (defaults to 1e100 - not used)
	double targetValue;

	/// maximum process time (defaults to 1e100 - not used)
	double maxSeconds;
};


///
/// \brief properties of the solution of a quadratic program
///
/// \par
/// The QpSolutionProperties structure collects statistics
/// about the approximate solutions found in a solver run.
/// It reports the reason for the iterative solver to stop,
/// which was set according to the QpStoppingCondition
/// structure. Furthermore it reports the solution accuracy,
/// the number of iterations, time elapsed, and the value
/// of the objective function in the reported solution.
///
struct QpSolutionProperties
{
	QpSolutionProperties()
	{
		type = QpNone;
		accuracy = 1e100;
		iterations = 0;
		value = 1e100;
		seconds = 0.0;
	}

	QpStopType type;						///< reason for the solver to stop
	double accuracy;						///< typically the maximal KKT violation
	unsigned long long iterations;			///< number of decomposition iterations
	double value;							///< value of the objective function
	double seconds;							///< training time
};

}
#endif
