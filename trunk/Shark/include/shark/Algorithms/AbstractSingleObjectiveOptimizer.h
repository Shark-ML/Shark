//===========================================================================
/*!
 * 
 *
 * \brief       AbstractSingleObjectiveOptimizer
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTSINGLEOBJECTIVEOPTIMIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTSINGLEOBJECTIVEOPTIMIZER_H

#include <shark/Algorithms/AbstractOptimizer.h>
#include <shark/Core/ResultSets.h>

namespace shark {
	///\brief Base class for all single objective optimizer
	///
	///This class is a spezialization of the AbstractOptimizer itnerface for the class of single objective optimizers. A single objective optimizer is an optimizer
	///which can only optimize functions with a single objective. This is the default case for most optimisation problems. 
	///the class requires the ObjectiveFunction to provide a feasible starting point. If this is not possible, a second version of init is provided where the starting point can be 
	///explicitely defined. 
	///The Return type of an SingleObjectiveOptimizer is the SingleObjectiveResultSet which is a struct returning the best value of the function and together with it's point.
	template<class PointType>
	class AbstractSingleObjectiveOptimizer: public AbstractOptimizer<PointType,double,SingleObjectiveResultSet<PointType> >{
	private:
		typedef AbstractOptimizer<PointType,double,SingleObjectiveResultSet<PointType> > base_type;
	public:
		typedef typename base_type::SearchPointType SearchPointType;
		typedef typename base_type::SolutionType SolutionType;
		typedef typename base_type::ResultType ResultType;
		typedef typename base_type::ObjectiveFunctionType ObjectiveFunctionType;

		///initializes the optimizer. The objectivefunction is required to provide a starting point, so CAN_PROPOSE_STARTING_POINT
		///must be set. If this is not the case, an exception is thrown
		virtual void init(ObjectiveFunctionType const& function ){
			if(!(function.features() & ObjectiveFunctionType::CAN_PROPOSE_STARTING_POINT))
				throw SHARKEXCEPTION( "[AbstractSingleObjectiveOptimizer::init] Objective function does not propose a starting point");
			RealVector startingPoint;
			function.proposeStartingPoint(startingPoint);
			init(function,startingPoint);
		}
		///initializes the optimizer using a predefined starting point
		virtual void init(ObjectiveFunctionType const& function, SearchPointType const& startingPoint)=0;
		///returns the current solution of the optimizer
		virtual const SolutionType& solution() const{
			return m_best;
		}

	protected:
		///current solution of the optimizer
		SolutionType m_best;
	};

}
#endif
