/*!
 * 
 *
 * \brief       Defines a base class for stopping criteria of optimization algorithms.
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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

#ifndef SHARK_ALGORITHMS_ABSTRACTSTOPPINGCRITERION_H
#define SHARK_ALGORITHMS_ABSTRACTSTOPPINGCRITERION_H

namespace shark{

/// \brief Base class for stopping criteria of optimization algorithms.
///
/// Each stopping criterion provides a stop method which takes as argument
/// the latest result of an optimization algorithm. This result is passed
/// as a ResultSetT, which is a template type of this interface. In actual
/// implementations, that type will usually hold the latest objective
/// function value reached by the optimizer on the training set, but may
/// also additionally encapsulate a performance value reached on a validation
/// set. The stopping criteria may then use this information to reach their
/// characteristic decision of whether or not stopping the optimization
/// process is indicated or not.
///
template<class ResultSetT>
class AbstractStoppingCriterion{
public:
	typedef ResultSetT ResultSet;
	///resets the internal state. call before a new trial
	virtual void reset() = 0;
	///after an iteration, updates the internal state and checks whether the algorithm should stop
	///@param set the result of the current iteration
	virtual bool stop(ResultSet const& set) = 0;
};

}

#endif
