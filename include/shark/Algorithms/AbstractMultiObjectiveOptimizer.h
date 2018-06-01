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
#ifndef SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEOPTIMIZER_H
#define SHARK_OBJECTIVEFUNCTIONS_ABSTRACTMULTIOBJECTIVEOPTIMIZER_H

#include <shark/Algorithms/AbstractOptimizer.h>
#include <shark/Core/ResultSets.h>

namespace shark {

///\defgroup multidirect Multi-objective Direct-Search optimizers
///\ingroup optimizers
/// Group of optimization algorithms that find a pareto front of the solutions on a multi-objective function
/// without using gradient information, only function values	
	
/// \brief base class for abstract multi-objective optimizers for arbitrary search spaces.
///
/// Models an abstract multi-objective optimizer for arbitrary search spaces. The objective space
/// is assumed to be \f$ \mathbb{R}^m\f$.
///
/// \tparam PointType The type of the points that make up the searchspace.
/// \ingroup optimizers
template<typename PointTypeT>
class AbstractMultiObjectiveOptimizer : 
public AbstractOptimizer<
	PointTypeT,
	RealVector,
	std::vector< ResultSet< PointTypeT, RealVector > > 
> {
private:
typedef AbstractOptimizer<
	PointTypeT,
	RealVector,
	std::vector< ResultSet< PointTypeT, RealVector > > 
> super;
public:
	typedef typename super::SearchPointType SearchPointType;
	typedef typename super::SolutionType SolutionType;
	typedef typename super::ObjectiveFunctionType ObjectiveFunctionType;

	/// \brief Accesses the current approximation of the Pareto-set and -front, respectively.
	/// \returns The current set of candidate solutions.
	SolutionType const& solution() const {
		return m_best;
	}

protected:
	SolutionType m_best; ///< The current Pareto-set/-front.
};

}
#endif
