//===========================================================================
/*!
 * 
 *
 * \brief       AbstractObjectiveFunction

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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTOBJECTIVEFUNCTION_H


#include <shark/Core/IConfigurable.h>
#include <shark/Core/INameable.h>
#include <shark/Core/Derivative.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Flags.h>
#include <shark/LinAlg/Base.h>
#include <shark/ObjectiveFunctions/AbstractConstraintHandler.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark {
/** \cond */
	/// \brief Models a vector space objective function storing the number of variables.
	/// 
	/// AbstractObjectiveFunction derives itself from this interface, when the search space type
	/// is a vector space and thus the notion of a number of variabls makes sense. This class is 
	/// otherwise not intended for direct use.
	class AbstractVectorSpaceObjectiveFunction {
	public:
		/// \brief Virtual d'tor.
		virtual ~AbstractVectorSpaceObjectiveFunction() {}

		/// \brief Accesses the number of variables
		virtual std::size_t numberOfVariables() const=0;
			
		virtual bool hasScalableDimensionality()const{
			return false;
		}

		/// \brief Adjusts the number of variables if the function is scalable.
		/// \param [in] numberOfVariables The new dimension.
		virtual void setNumberOfVariables( std::size_t numberOfVariables ){
			throw SHARKEXCEPTION("dimensionality of function is not scalable");
		}
	};
/** \endcond */

/// \brief Super class of all objective functions for optimization and learning.

/// \par
/// The AbstractObjectiveFunction template class is the most general
/// interface for a function to be minimized or maximized by an
/// optimizer. It subsumes many more specialized classes,
/// ranging from classical test problems in evolutionary algorithms to
/// data-dependent objective functions in supervised learning. This
/// interface allows all general purpose optimization procedures to be
/// used as model training algorithms in a learning task, with
/// applications ranging from training of neural networks to direct
/// policy search in reinforcement learning.

/// AbstractObjectiveFunction offers a rich interface to support
/// different types of optimizers. Since not every objective function meets
/// every requirement, a flagsystem exists which tells the optimizer
/// which Features are available. These are:
/// HAS_VALUE: The function can be evaluated. If not set, evalDerivative returns a meaningless
/// value (for example std::numeric_limits<double>::quiet_nan());
/// HAS_FIRST_DERIVATIVE: evalDerivative can be called for the FirstOrderDerivative.
/// The Derivative is defined and as exact as possible;
/// HAS_SECOND_DERIVATIVE: evalDerivative can be called for the second derivative.
/// It is defined and non-zero;
/// IS_CONSTRAINED_FEATURE: The function has constraints and isFeasible might return false;
/// CAN_PROPOSE_STARTING_POINT: the function can return a possibly randomized starting point;
/// CAN_PROVIDE_CLOSEST_FEASIBLE: if the function is constrained, closest feasible can be
/// called to construct a feasible point.

/// Calling the derivatives, proposeStartingPoint or closestFeasible when the flags are not set
/// will throw an exception.
/// The features can be queried using the method features() as in
/// if(!(f.features()&Function::HAS_VALUE))

/// \tparam SearchSpaceT The search space the function is defined upon.
/// \tparam ResultT The objective space the function is defined upon.
template <typename SearchSpaceT, typename ResultT>
class AbstractObjectiveFunction : public IConfigurable, 
	public INameable, 
	/** \cond */
	public boost::mpl::if_c< SearchSpaceT::IS_VECTOR_SPACE, AbstractVectorSpaceObjectiveFunction, boost::mpl::void_ >::type
	/** \endcond */{
public:
	typedef SearchSpaceT SearchSpaceType;
	typedef typename SearchSpaceT::PointType SearchPointType;
	typedef ResultT ResultType;

	typedef SearchPointType FirstOrderDerivative;
	typedef TypedSecondOrderDerivative<SearchPointType,RealMatrix> SecondOrderDerivative;

	/// \brief List of features that are supported by an implementation.
	enum Feature {
		HAS_VALUE						 =  1, ///< The function can be evaluated and evalDerivative returns a meaningless value (for example std::numeric_limits<double>::quiet_nan()).
		HAS_FIRST_DERIVATIVE             =  2, ///< The method evalDerivative is implemented for the first derivative and returns a sensible value.
		HAS_SECOND_DERIVATIVE            =  4, ///< The method evalDerivative is implemented for the second derivative and returns a sensible value.
		CAN_PROPOSE_STARTING_POINT       = 8, ///< The function can propose a sensible starting point to search algorithms.
		IS_CONSTRAINED_FEATURE           =  16, ///< The objective function is constrained.
		HAS_CONSTRAINT_HANDLER           =  32, ///< The constraints are governed by a constraint handler which can be queried by getConstraintHandler()
		CAN_PROVIDE_CLOSEST_FEASIBLE     = 64,	///< If the function is constrained, the method closestFeasible is implemented and returns a "repaired" solution.
		IS_THREAD_SAFE     = 128	///< can eval or evalDerivative be called in parallel?
	};

	/// This statement declares the member m_features. See Core/Flags.h for details.
	SHARK_FEATURE_INTERFACE;
	
	/// \brief returns whether this function can calculate it's function value
	bool hasValue()const{
		return m_features & HAS_VALUE;
	}
	
	/// \brief returns whether this function can calculate the first derivative
	bool hasFirstDerivative()const{
		return m_features & HAS_FIRST_DERIVATIVE;
	}
	
	/// \brief returns whether this function can calculate the second derivative
	bool hasSecondDerivative()const{
		return m_features & HAS_SECOND_DERIVATIVE;
	}
	
	/// \brief returns whether this function can propose a starting point.
	bool canProposeStartingPoint()const{
		return m_features & CAN_PROPOSE_STARTING_POINT;
	}
	
	/// \brief returns whether this function can return 
	bool isConstrained()const{
		return m_features & IS_CONSTRAINED_FEATURE;
	}
	
	/// \brief returns whether this function can return 
	bool hasConstraintHandler()const{
		return m_features & HAS_CONSTRAINT_HANDLER;
	}
	
	/// \brief Returns whether this function can calculate thee closest feasible to an infeasible point.
	bool canProvideClosestFeasible()const{
		return m_features & CAN_PROVIDE_CLOSEST_FEASIBLE;
	}
	
	/// \brief Returns true, when the function can be usd in parallel threads.
	bool isThreadSafe()const{
		return m_features & IS_THREAD_SAFE;
	}

	/// \brief Default ctor.
	AbstractObjectiveFunction():m_evaluationCounter(0) {
	    m_features |=HAS_VALUE;
	}
	/// \brief Virtual destructor
	virtual ~AbstractObjectiveFunction() {}

	virtual void configure( const PropertyTree & node ) {
		(void) node;
	}

	virtual void init() {}
		
	virtual std::size_t numberOfObjectives() const{
		return 1;
	}
	virtual bool hasScalableObjectives()const{
		return false;
	}

	/// \brief Adjusts the number of objectives if the function is scalable.
	/// \param numberOfObjectives The new number of objectives to optimize for.
	virtual void setNumberOfObjectives( std::size_t numberOfObjectives ){
		throw SHARKEXCEPTION("dimensionality of function is not scaleable");
	}
	
		
	/// \brief Accesses the evaluation counter of the function.
	std::size_t evaluationCounter() const {
		return m_evaluationCounter;
	}
	
	/// \brief Returns the constraint handler of the function if it has one.
	///
	/// If the function does not offer a constraint handler, an exception is thrown.
	AbstractConstraintHandler<SearchPointType> const& getConstraintHandler()const{
		if(m_constraintHandler == NULL)
			throw SHARKEXCEPTION("Objective Function does not have an constraint handler!");
		return *m_constraintHandler;
	}

	/// \brief Tests whether a point in SearchSpace is feasible, e.g., whether the constraints are fulfilled.
	/// \param [in] input The point to be tested for feasibility.
	/// \return true if the point is feasible, false otherwise.
	virtual bool isFeasible( const SearchPointType & input) const {
		if(hasConstraintHandler()) return getConstraintHandler().isFeasible(input);
		if(isConstrained())
			throw SHARKEXCEPTION("[AbstractObjectiveFunction::isFasible] not overwritten, even though function is constrained");
		return true;
	}

	/// \brief If supported, the supplied point is repaired such that it satisfies all of the function's constraints.
	/// 
	/// \param [in,out] input The point to be repaired.
	/// 
	/// \throws FeatureNotAvailableException in the default implementation.
	virtual void closestFeasible( SearchPointType & input ) const {
		if(!isConstrained()) return;
		if(hasConstraintHandler()) return getConstraintHandler().closestFeasible(input);
		SHARK_FEATURE_EXCEPTION(CAN_PROVIDE_CLOSEST_FEASIBLE);
	}

	///  \brief Proposes a starting point in the feasible search space of the function.
	/// 
	///  \param [out] startingPoint The starting point is placed here.
	///  \throws FeatureNotAvailableException in the default implementation
	///  and if a function does not support this feature.
	virtual void proposeStartingPoint( SearchPointType & startingPoint )const {
		if(hasConstraintHandler()&& getConstraintHandler().canGenerateRandomPoint())
			getConstraintHandler().generateRandomPoint(startingPoint);
		else{
			SHARK_FEATURE_EXCEPTION(CAN_PROPOSE_STARTING_POINT);
		}
	}

	///  \brief Evaluates the objective function for the supplied argument.
	///  \param [in] input The argument for which the function shall be evaluated.
	///  \return The result of evaluating the function for the supplied argument.
	///  \throws FeatureNotAvailableException in the default implementation
	///  and if a function does not support this feature.
	virtual ResultType eval( const SearchPointType & input )const {
		SHARK_FEATURE_EXCEPTION(HAS_VALUE);
	}

	/// \brief Evaluates the function. Useful together with STL-Algorithms like std::transform.
	ResultType operator()( const SearchPointType & input ) const {
		return eval(input);
	}

	/// \brief Evaluates the objective function and calculates its gradient.
	/// \param [in] input The argument to eval the function for.
	/// \param [out] derivative The derivate is placed here.
	/// \return The result of evaluating the function for the supplied argument.
	/// \throws FeatureNotAvailableException in the default implementation
	/// and if a function does not support this feature.
	virtual ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative )const {
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_DERIVATIVE);
	}

	/// \brief Evaluates the objective function and calculates its gradient.
	/// \param [in] input The argument to eval the function for.
	/// \param [out] derivative The derivate and the Hessian are placed here.
	/// \return The result of evaluating the function for the supplied argument.
	/// \throws FeatureNotAvailableException in the default implementation
	/// and if a function does not support this feature.
	virtual ResultType evalDerivative( const SearchPointType & input, SecondOrderDerivative & derivative )const {
		SHARK_FEATURE_EXCEPTION(HAS_SECOND_DERIVATIVE);
	}

protected:
	mutable std::size_t m_evaluationCounter; ///< Evaluation counter, default value: 0.
	AbstractConstraintHandler<SearchPointType> const* m_constraintHandler;
	
	/// \brief helper function which is called to announce the presence of an constraint handler.
	///
	/// This function quries the propabilities of the handler and sts up the flags accordingly
	void announceConstraintHandler(AbstractConstraintHandler<SearchPointType> const* handler){
		SHARK_CHECK(handler != NULL, "[AbstractObjectiveFunction::AnnounceConstraintHandler] Handler is not allowed to be NULL");
		m_constraintHandler = handler;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		if(handler->canGenerateRandomPoint())
			m_features |=CAN_PROPOSE_STARTING_POINT;
		if(handler->canProvideClosestFeasible())
			m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
	}
};

typedef AbstractObjectiveFunction< VectorSpace< double >, double > SingleObjectiveFunction;
typedef AbstractObjectiveFunction< VectorSpace< double >, RealVector > MultiObjectiveFunction;

}

#endif
