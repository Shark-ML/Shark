//===========================================================================
/*!
 * 
 *
 * \brief       AbstractOptimizer
 * 
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
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
//===========================================================================
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTOPTIMIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTOPTIMIZER_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>

namespace shark {

/**
* \brief An optimizer that optimizes general objective functions
* 
* After construction and configurationg the optimizer, init() is called with the objective function
* to be used. After that step() can be called until the required solution is found. The solution can be queried
* using solution(). The type of the solution depends on the optimisation problem at hand.
* It is allowed to add constrains on the features the objective function needs to offer
* 
* These are:
*	- REQUIRES_VALUE: The function is evaluated to use the optimizer and
*	  the HAS_VALUE-flag must be set
*	- REQUIRES_FIRST_DERIVATIVE: The first derivative needs to be evaluated and
*	- HAS_FIRST_DERIVATIVE must be set
*	- REQUIRES_SECOND_DERIVATIVE: The second derivative needs to be evaluated and
*	- HAS_SECOND_DERIVATIVE must be set
*	- CAN_SOLVE_CONSTRAINED: The optimizer can solve functions which are constrained and
*	  where the IS_CONSTRAINED_FEATURE is set.
*	- REQUIRES_CLOSEST_FEASIBLE: If the function is constrained, it must offer a way to
*	construct the closest feasible point and
*	- CAN_PROVIDE_CLOSEST_FEASIBLE must be set
* 
* Also when init() is called as offered by the AbstractOptimizer interface, the function
* is required to have the CAN_PROPOSE_STARTING_POINT flag.
* 
* \tparam PointType The type of search space the optimizer works upon.
* \tparam ResultT The objective space the optimizer works upon.
* \tparam SolutionTypeT The type of the final solution.
*/
template <typename PointType, typename ResultT, typename SolutionTypeT>
class AbstractOptimizer : public INameable, public ISerializable {
public:
	typedef PointType SearchPointType;
	typedef ResultT ResultType;
	typedef SolutionTypeT SolutionType;
	typedef AbstractObjectiveFunction<PointType,ResultType> ObjectiveFunctionType;

	/**
	* \brief Models features that the optimizer requires from the objective function.
	* \sa AbstractObjectiveFunction
	*/
	enum Feature {
		REQUIRES_VALUE          	=  1,
		REQUIRES_FIRST_DERIVATIVE	=  2,
		REQUIRES_SECOND_DERIVATIVE	=  4,
		CAN_SOLVE_CONSTRAINED           =  8,
		REQUIRES_CLOSEST_FEASIBLE       = 16
	};

	SHARK_FEATURE_INTERFACE;
	
	bool requiresValue()const{
		return features()& REQUIRES_VALUE;
	}
	
	bool requiresFirstDerivative()const{
		return features()& REQUIRES_FIRST_DERIVATIVE;
	}
	bool requiresSecondDerivative()const{
		return features()& REQUIRES_SECOND_DERIVATIVE;
	}
	bool canSolveConstrained()const{
		return features()& CAN_SOLVE_CONSTRAINED;
	}
	bool requiresClosestFeasible()const{
		return features()& REQUIRES_CLOSEST_FEASIBLE;
	}

	/**
	* \brief Empty virtual d'tor.
	*/
	virtual ~AbstractOptimizer() {}

	/**
	* \brief Initialize the optimizer for the supplied objective function.
	* \param [in] function The objective function to initialize for.
	*/
	virtual void init( ObjectiveFunctionType const& function ) = 0;

	/**
	* \brief Carry out one step of the optimizer for the supplied objective function.
	* \param [in] function The objective function to initialize for.
	*/
	virtual void step( ObjectiveFunctionType const& function ) = 0;

	/**
	* \brief Accesses the best solution obtained so far. 
	* \returns An immutable reference to the best solution obtained so far.
	*/
	virtual SolutionType const& solution() const = 0; //mt_hint: try accessing this thing via solution().point and solution().value..

protected:
	/**
	* \brief Convenience function that checks whether the features of the supplied objective function match with the required features of the optimizer.
	* \param [in] objectiveFunction The function to match with.
	* \throws shark::Exception
	*/
	void checkFeatures (const ObjectiveFunctionType & objectiveFunction){
		//test first derivative
		if( (m_features & REQUIRES_FIRST_DERIVATIVE) &
			!(objectiveFunction.features() & ObjectiveFunctionType::HAS_FIRST_DERIVATIVE)
		)throw SHARKEXCEPTION("[ "+name()+" ] requires first derivative");
		//test second derivative
		if( (m_features & REQUIRES_SECOND_DERIVATIVE) &
			!(objectiveFunction.features() & ObjectiveFunctionType::HAS_SECOND_DERIVATIVE)
		)throw SHARKEXCEPTION("[ "+name()+" ] requires second derivative");

		//test whether the function can be evaluated
		if( (m_features & REQUIRES_VALUE) &
			!(objectiveFunction.features() & ObjectiveFunctionType::HAS_VALUE)
		)throw SHARKEXCEPTION("[ "+name()+" ] requires the value of the function");

		//test for constrains
		if( !(m_features & CAN_SOLVE_CONSTRAINED) &
			(objectiveFunction.features() & ObjectiveFunctionType::IS_CONSTRAINED_FEATURE)
		)throw SHARKEXCEPTION("[ "+name()+" ] can not solve constrained functions");

		//test for closest feasible in constrained functions
		if( (objectiveFunction.features() & ObjectiveFunctionType::IS_CONSTRAINED_FEATURE) &
			!(objectiveFunction.features() & ObjectiveFunctionType::CAN_PROVIDE_CLOSEST_FEASIBLE) &
			(m_features & REQUIRES_CLOSEST_FEASIBLE)
		)throw SHARKEXCEPTION("[ "+name()+" ] requires closest feasible for constrained functions");
	}
};

}

#endif // SHARK_CORE_ABSTRACTOPTIMIZER_H
