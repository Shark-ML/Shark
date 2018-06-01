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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTSINGLEOBJECTIVEOPTIMIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTSINGLEOBJECTIVEOPTIMIZER_H

#include <shark/Algorithms/AbstractOptimizer.h>
#include <shark/Core/ResultSets.h>

namespace shark {
	
	///\defgroup gradientopt Gradient-based Single objective optimizers
	///\ingroup optimizers
	/// Gradient-Based optimizers use the gradient of an objective function to find a local minimum in the search space.
	/// If a function is not convex this local optimum might not be the global one.
	
	///\defgroup singledirect Single-objective Direct-Search optimizers
	///\ingroup optimizers
	/// Group of optimization algorithms that find the local optimum of a function without using gradient information, only
	/// the function values are used.
	
	///\brief Base class for all single objective optimizer
	///
	/// This class is a spezialization of the AbstractOptimizer itnerface for the class of single objective optimizers. A single objective optimizer is an optimizer
	/// which can only optimize functions with a single objective. This is the default case for most optimisation problems. 
	/// the class requires the ObjectiveFunction to provide a feasible starting point. If this is not possible, a second version of init is provided where the starting point can be 
	/// explicitely defined. 
	/// The Return type of an SingleObjectiveOptimizer is the SingleObjectiveResultSet which is a struct returning the best value of the function and together with it's point.
	/// \ingroup optimizers
	template<class PointType>
	class AbstractSingleObjectiveOptimizer: public AbstractOptimizer<PointType,double,SingleObjectiveResultSet<PointType> >{
	private:
		typedef AbstractOptimizer<PointType,double,SingleObjectiveResultSet<PointType> > base_type;
	public:
		typedef typename base_type::SearchPointType SearchPointType;
		typedef typename base_type::SolutionType SolutionType;
		typedef typename base_type::ResultType ResultType;
		typedef typename base_type::ObjectiveFunctionType ObjectiveFunctionType;

		///\brief By default most single objective optimizers only require a single point
		std::size_t numInitPoints() const{
			return 1;
		}
		
		using base_type::init;
		
		/// \brief Initialize the optimizer for the supplied objective function using a set of initialisation points
		///
		/// The default implementation picks either the first point in the set, or if it is enmpty, trys
		/// to generate one from the function.
		///
		/// Be aware that function.init() has to be called before calling this function!
		///
		/// \param [in] function The objective function to initialize for.
		/// \param [in] initPoints points used for initialisation. Should be at least numInitPoints().
		virtual void init( ObjectiveFunctionType const& function, std::vector<SearchPointType> const& initPoints ){
			if(initPoints.empty())
				init(function);
			else
				init(function,initPoints[0]);
		}
		
		///initializes the optimizer using a predefined starting point
		virtual void init(ObjectiveFunctionType const& function, SearchPointType const& startingPoint)=0;
		///returns the current solution of the optimizer
		virtual const SolutionType& solution() const{
			return m_best;
		}

	protected:
		
		SolutionType m_best; ///<Current solution of the optimizer
	};

}
#endif
