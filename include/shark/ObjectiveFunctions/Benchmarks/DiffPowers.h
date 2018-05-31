/*!
 * 
 *
 * \brief       -
 *
 * \author      -
 * \date        -
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DIFFPOWERS_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_DIFFPOWERS_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>


/// \brief Different powers objective function.
///
/// each dimension is of the form x^a for varying a. This function is ill-conditioned close to the optimum.
/// \ingroup benchmarks
namespace shark {namespace benchmarks{
struct DiffPowers : public SingleObjectiveFunction {

	DiffPowers(std::size_t numberOfVariables = 5) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DiffPowers"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::uni(*mep_rng, 0,1);
		}
		return x;
	}

	double eval( const SearchPointType & p ) const {
		m_evaluationCounter++;
		double sum = 0;
		for( std::size_t i = 0; i < p.size(); i++ ){
			sum += std::pow( std::abs( p( i ) ), 2. + (10.*i) / (p.size() - 1.) );
		}
		return sum;
	}
private:
	std::size_t m_numberOfVariables;
};
}}

#endif
