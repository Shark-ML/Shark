//===========================================================================
/*!
 * 
 *
 * \brief       AbstractObjectiveFunction
 * \file
 * 
 *
 * \author      T.Voss, T. Glasmachers, O.Krause
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#include <shark/Core/INameable.h>
#include <shark/Core/Exception.h>
#include <shark/Core/Flags.h>
#include <shark/LinAlg/Base.h>
#include <shark/ObjectiveFunctions/AbstractConstraintHandler.h>

namespace shark {
	
/// \defgroup objfunctions Objective functions
/// \brief Objective functions for optimization.
///
/// In shark, the learning problem is phrased as an objective function which is then optimized using an \ref optimizer. This allows
/// to test and develop algorithms using \ref benchmarks independent of the problem to solve.

/// \brief Super class of all objective functions for optimization and learning.
///
/// \par
/// The AbstractObjectiveFunction template class is the most general
/// interface for a function to be minimized by an
/// optimizer. It subsumes many more specialized classes,
/// ranging from classical test problems in evolutionary algorithms to
/// data-dependent objective functions in supervised learning. This
/// interface allows all general purpose optimization procedures to be
/// used as model training algorithms in a learning task, with
/// applications ranging from training of neural networks to direct
/// policy search in reinforcement learning.
///
/// AbstractObjectiveFunction offers a rich interface to support
/// different types of optimizers. Since not every objective function meets
/// every requirement, a flag system exists which tells the optimizer
/// which features are available. These are:
/// HAS_VALUE: The function can be evaluated. If not set, evalDerivative returns a meaningless
/// value (for example std::numeric_limits<double>::quiet_nan());
/// HAS_FIRST_DERIVATIVE: evalDerivative can be called for the FirstOrderDerivative.
/// The Derivative is defined and as exact as possible;
/// HAS_SECOND_DERIVATIVE: evalDerivative can be called for the second derivative.
/// IS_CONSTRAINED_FEATURE: The function has constraints and isFeasible might return false;
/// CAN_PROPOSE_STARTING_POINT: the function can return a possibly randomized starting point;
/// CAN_PROVIDE_CLOSEST_FEASIBLE: if the function is constrained, closest feasible can be
/// called to construct a feasible point.
///
/// In the single objective case, the shark convention is to return a double value, while in
/// Multi objective optimization a RealVector is returned with an entry for every objective.
/// Moreoever, derivatives in the single objective case are RealVectors, while they are 
/// RealMatrix in the multi-objective case (i.e. the jacobian of the function).
///
/// Calling the derivatives, proposeStartingPoint or closestFeasible when the flags are not set
/// will throw an exception.
/// The features can be queried using the method features() as in
/// if(!(f.features()&Function::HAS_VALUE))
///
/// \ingroup objfunctions
/// \tparam PointType The search space the function is defined upon.
/// \tparam ResultT The objective space the function is defined upon.
template <typename PointType, typename ResultT>
class AbstractObjectiveFunction :  public INameable{
public:
	typedef PointType SearchPointType;
	typedef ResultT ResultType;

	//if the result type is not an arithmetic type, we assume it is a vector-type->multi objective optimization
	typedef typename boost::mpl::if_<
		std::is_arithmetic<ResultT>,
		SearchPointType,
		RealMatrix
	>::type FirstOrderDerivative;

	struct SecondOrderDerivative {
		FirstOrderDerivative gradient;
		RealMatrix hessian;
	};

	/// \brief List of features that are supported by an implementation.
	enum Feature {
		HAS_VALUE						 =  1, ///< The function can be evaluated and evalDerivative returns a meaningless value (for example std::numeric_limits<double>::quiet_nan()).
		HAS_FIRST_DERIVATIVE             =  2, ///< The method evalDerivative is implemented for the first derivative and returns a sensible value.
		HAS_SECOND_DERIVATIVE            =  4, ///< The method evalDerivative is implemented for the second derivative and returns a sensible value.
		CAN_PROPOSE_STARTING_POINT       = 8, ///< The function can propose a sensible starting point to search algorithms.
		IS_CONSTRAINED_FEATURE           =  16, ///< The objective function is constrained.
		HAS_CONSTRAINT_HANDLER           =  32, ///< The constraints are governed by a constraint handler which can be queried by getConstraintHandler()
		CAN_PROVIDE_CLOSEST_FEASIBLE     = 64,	///< If the function is constrained, the method closestFeasible is implemented and returns a "repaired" solution.
		IS_THREAD_SAFE     = 128,	///< can eval or evalDerivative be called in parallel?
		IS_NOISY     = 256	///< The function value is perturbed by some kind of noise
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
	
	/// \brief Returns true, when the function can be usd in parallel threads.
	bool isNoisy()const{
		return m_features & IS_NOISY;
	}

	/// \brief Default ctor.
	AbstractObjectiveFunction():m_evaluationCounter(0), mep_rng(&random::globalRng){
	    m_features |=HAS_VALUE;
	}
	/// \brief Virtual destructor
	virtual ~AbstractObjectiveFunction() {}

	virtual void init() {
		m_evaluationCounter=0;
	}
	
	///\brief Sets the Rng used by the objective function.
	///
	/// Objective functions need random numbers for different tasks,
	/// e.g. to provide a first starting point or for example
	/// mini batch learning where batches are chosen randomly. 
	/// By default, shark::random::globalRng is used.
	/// In a multi-threaded environment this might not be safe as 
	/// the Rng is not thread safe. In this case, every thread should use its
	/// own Rng.
	void setRng(random::rng_type* rng){
		mep_rng = rng;
	}
	
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
		SHARK_RUNTIME_CHECK(m_constraintHandler, "Objective Function does not have an constraint handler!");
		return *m_constraintHandler;
	}

	/// \brief Tests whether a point in SearchSpace is feasible, e.g., whether the constraints are fulfilled.
	/// \param [in] input The point to be tested for feasibility.
	/// \return true if the point is feasible, false otherwise.
	virtual bool isFeasible( const SearchPointType & input) const {
		if(hasConstraintHandler()) return getConstraintHandler().isFeasible(input);
		SHARK_RUNTIME_CHECK(!isConstrained(), "Not overwritten, even though function is constrained");
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
	///  \return The generated starting point.
	///  \throws FeatureNotAvailableException in the default implementation
	///  and if a function does not support this feature.
	virtual SearchPointType proposeStartingPoint()const {
		if(hasConstraintHandler()&& getConstraintHandler().canGenerateRandomPoint()){
			SearchPointType startingPoint;
			getConstraintHandler().generateRandomPoint(*mep_rng, startingPoint);
			return startingPoint;
		}
		else{
			SHARK_FEATURE_EXCEPTION(CAN_PROPOSE_STARTING_POINT);
		}
	}

	///  \brief Evaluates the objective function for the supplied argument.
	///  \param [in] input The argument for which the function shall be evaluated.
	///  \return The result of evaluating the function for the supplied argument.
	///  \throws FeatureNotAvailableException in the default implementation
	///  and if a function does not support this feature.
	virtual ResultType eval( SearchPointType const& input )const {
		SHARK_FEATURE_EXCEPTION(HAS_VALUE);
	}

	/// \brief Evaluates the function. Useful together with STL-Algorithms like std::transform.
	ResultType operator()( SearchPointType const& input ) const {
		return eval(input);
	}

	/// \brief Evaluates the objective function and calculates its gradient.
	/// \param [in] input The argument to eval the function for.
	/// \param [out] derivative The derivate is placed here.
	/// \return The result of evaluating the function for the supplied argument.
	/// \throws FeatureNotAvailableException in the default implementation
	/// and if a function does not support this feature.
	virtual ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative )const {
		SHARK_FEATURE_EXCEPTION(HAS_FIRST_DERIVATIVE);
	}

	/// \brief Evaluates the objective function and calculates its gradient.
	/// \param [in] input The argument to eval the function for.
	/// \param [out] derivative The derivate and the Hessian are placed here.
	/// \return The result of evaluating the function for the supplied argument.
	/// \throws FeatureNotAvailableException in the default implementation
	/// and if a function does not support this feature.
	virtual ResultType evalDerivative( SearchPointType const& input, SecondOrderDerivative & derivative )const {
		SHARK_FEATURE_EXCEPTION(HAS_SECOND_DERIVATIVE);
	}

protected:
	mutable std::size_t m_evaluationCounter; ///< Evaluation counter, default value: 0.
	AbstractConstraintHandler<SearchPointType> const* m_constraintHandler;
	random::rng_type* mep_rng;
	
	/// \brief helper function which is called to announce the presence of an constraint handler.
	///
	/// This function quries the propabilities of the handler and sts up the flags accordingly
	void announceConstraintHandler(AbstractConstraintHandler<SearchPointType> const* handler){
		SHARK_RUNTIME_CHECK(handler, "[AbstractObjectiveFunction::AnnounceConstraintHandler] Handler is not allowed to be NULL");
		m_constraintHandler = handler;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= HAS_CONSTRAINT_HANDLER;
		if(handler->canGenerateRandomPoint())
			m_features |=CAN_PROPOSE_STARTING_POINT;
		if(handler->canProvideClosestFeasible())
			m_features |= CAN_PROVIDE_CLOSEST_FEASIBLE;
	}
};

typedef AbstractObjectiveFunction< RealVector, double > SingleObjectiveFunction;
typedef AbstractObjectiveFunction< RealVector, RealVector > MultiObjectiveFunction;

}

#endif
