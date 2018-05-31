/*!
 * 
 *
 * \brief       Convex benchmark function.
 * 
 *
 * \author      T. Voss
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SCHWEFEL_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SCHWEFEL_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Convex benchmark function.
*  \ingroup benchmarks
 */
struct Schwefel : public SingleObjectiveFunction {
	
	Schwefel(std::size_t numberOfVariables = 5):m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Schwefel"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}

	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	SearchPointType proposeStartingPoint() const {
		RealVector x(numberOfVariables());

		for (std::size_t i = 0; i < x.size(); i++) {
			x(i) = random::gauss(*mep_rng, 0,1);
		}
		return x;
	}

	double eval(const SearchPointType &p) const {
		m_evaluationCounter++;
		double value = 0;
		double sum= 0;
		for(std::size_t i = 0; i != m_numberOfVariables; ++i){
			sum+= p(i);
			value+=sqr(sum);
		}
		return value;
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
