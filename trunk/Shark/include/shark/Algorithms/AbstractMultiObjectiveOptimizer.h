/*!
 * 
 *
 * \brief       AbstractMultiObjectiveOptimizer
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEOPTIMIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEOPTIMIZER_H

#include <shark/Algorithms/AbstractOptimizer.h>
#include <shark/Core/ResultSets.h>

namespace shark {

/**
 * \brief base class for abstract multi-objective optimizers for arbitrary search spaces.
 *
 * Models an abstract multi-objective optimizer for arbitrary search spaces. The objective space
 * is assumed to be \f$ \mathbb{R}^m\f$.
 *
 * \tparam SearchSpace The search space of the optimizer.
 */
template<typename SearchSpace>
class AbstractMultiObjectiveOptimizer : 
public AbstractOptimizer<
	SearchSpace,
	shark::RealVector,
	std::vector< ResultSet< typename SearchSpace::PointType, shark::RealVector > > 
> {
private:
typedef AbstractOptimizer<
	SearchSpace,
	shark::RealVector,
	std::vector< ResultSet< typename SearchSpace::PointType, shark::RealVector > > 
> super;
public:
	typedef typename super::SearchSpaceType SearchSpaceType;
	typedef typename super::SearchPointType SearchPointType;
	typedef typename super::SolutionSetType SolutionSetType;
	typedef typename super::ResultType ResultType;
	typedef typename super::ObjectiveFunctionType ObjectiveFunctionType;

	/**
	* \brief Virtual empty d'tor.
	*/
	virtual ~AbstractMultiObjectiveOptimizer() {}

	/**
	* \brief Initializes the optimizer for the supplied objective function.
	*
	* Tries to sample an initial starting point. If the function does not
	* implement this feature, an exception is thrown. Otherwise, the call is dispatched
	* to the pure-virtual function.
	*
	* \param function The function to be initialized for.
	* \throws shark::Exception if the function does not feature the proposal of starting points.
	*/
	virtual void init( const ObjectiveFunctionType & function ) {
		if(!(function.features() & ObjectiveFunctionType::CAN_PROPOSE_STARTING_POINT))
			throw SHARKEXCEPTION( "Objective function does not propose a starting point");
		std::vector<RealVector> startingPoints(1);
		function.proposeStartingPoint(startingPoints[0]);
		init(function,startingPoints);
	}

	/**
	* \brief Optimizer-specific init-function. Needs to be implemented by subclasses.
	* \param [in] function The function to initialize the optimizer for.
	* \param [in] startingPoints An initial population of points
	*/
	virtual void init( 
		ObjectiveFunctionType const& function, 
		std::vector<SearchPointType> const& startingPoints
	) = 0;

	/**
	* \brief Accesses the current approximation of the Pareto-set and -front, respectively.
	* \returns The current set of candidate solutions.
	*/
	const SolutionSetType & solution() const {
		return m_best;
	}

protected:
	SolutionSetType m_best; ///< The current Pareto-set/-front.
};

}

#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {


/// \brief Type erasure to integrate Optimizer adhering to the concept of a 
/// multi-objective optimizer with the inheritance hierarchy of AbstractOptimizer.
///
/// \tparam SearchSpace The search space.
/// \tparam Optimizer The optimizer adhering to the concept of a multi-objective optimizer.
template<typename SearchSpace, typename Optimizer>
class TypeErasedMultiObjectiveOptimizer
: public AbstractMultiObjectiveOptimizer<SearchSpace>
, public Optimizer 
{
public:
	typedef AbstractMultiObjectiveOptimizer<SearchSpace> super;
	typedef Optimizer OptimizerBase;
	typedef typename OptimizerBase::SolutionSetType SolutionSetType;

	/// \brief Default c'tor, initializes the name of the optimizer.
	TypeErasedMultiObjectiveOptimizer() {
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return Optimizer::name(); }

	using super::init;

	/// \brief Configures the optimizer with the supplied property tree.
	/// \param [in] node The root of the property tree.
	void configure( const PropertyTree & node ) {
	OptimizerBase::configure( node );
	}

	///\brief Optimizer-specific init-function. Dispatched to the optimizer provided as template argument to this class.
	///\param [in] function The function to initialize the optimizer for.
	///\param [in] startingPoint An initial point sampled from the function.
	void init( const typename super::ObjectiveFunctionType & function, const typename super::SearchPointType & startingPoint ) {
		OptimizerBase::init( function, startingPoint );
	}

	/// \brief Optimizer-specific init-function. Dispatched to the optimizer provided as template argument to this class.
	/// \param [in] function The function to initialize the optimizer for.
	void step( const typename super::ObjectiveFunctionType & function ) {
		super::m_best = OptimizerBase::step( function );
	}
};
}

#endif