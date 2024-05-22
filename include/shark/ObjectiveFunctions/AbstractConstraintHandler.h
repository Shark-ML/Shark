//===========================================================================
/*!
 * 
 *
 * \brief       Base class for constraints.
 *  \file
 *
 * \author      O.Krause
 * \date        2013
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <https://shark-ml.github.io/Shark/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTCONSTRAINTHANDLER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTCONSTRAINTHANDLER_H

#include <shark/Core/Exception.h>
#include <shark/Core/Flags.h>
#include <shark/Core/Random.h>

namespace shark{

/// \defgroup constraint_handling Constraint Handling
/// \ingroup objfunctions
/// \brief Objects for handling constraints
///
	
/// \brief Implements the base class for constraint handling.
///
/// A constraint handler provides information about the feasible region of a constrained optimization problem.
/// In the minimum it checks whether a point is feasible, or what the next fasible point would be.
/// \ingroup constraint_handling
template<class SearchPointType>
class AbstractConstraintHandler{
public:
	enum Feature {
		CAN_PROVIDE_CLOSEST_FEASIBLE     = 1,	///< The constraint handler can provide a close feasible point to an infeasible one
		IS_BOX_CONSTRAINED = 2,  ///< The constraint handler is an instance of BoxConstraintHandler
		CAN_GENERATE_RANDOM_POINT = 4  ///< The ConstraintHandler can generate a random point inside the feasible region
	};
	SHARK_FEATURE_INTERFACE;
	
	virtual ~AbstractConstraintHandler(){}
		
	/// \brief Returns whether this function can calculate the closest feasible to an infeasible point.
	bool canProvideClosestFeasible()const{
		return m_features & CAN_PROVIDE_CLOSEST_FEASIBLE;
	}
	
	/// \brief Returns whether this function is an instance of BoxConstraintHandler
	bool isBoxConstrained()const{
		return m_features &IS_BOX_CONSTRAINED;
	}
	/// \brief Returns whether this function is an instance of BoxConstraintHandler
	bool canGenerateRandomPoint()const{
		return m_features & CAN_GENERATE_RANDOM_POINT;
	}
	
	/// \brief If supported, generates a random point inside the feasible region.
	///
	/// \param rng The random number generator used for generating the point
	/// \param startingPoint The proposed point
	virtual void generateRandomPoint( random::rng_type& rng, SearchPointType & startingPoint )const {
		SHARK_FEATURE_EXCEPTION(CAN_GENERATE_RANDOM_POINT);
	}
	
	/// \brief Returns true if the point is in the feasible Region.
	///
	/// This function must be implemented by a ConstraintHandler
	virtual bool isFeasible(SearchPointType const&)const = 0;
	virtual void closestFeasible(SearchPointType& )const{
		SHARK_FEATURE_EXCEPTION(CAN_PROVIDE_CLOSEST_FEASIBLE );
	}
	
};
}
#endif
