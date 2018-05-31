/*!
 * 
 *
 * \brief       Convex quadratic benchmark function.
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SPHERE_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_SPHERE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Convex quadratic benchmark function.
*  \ingroup benchmarks
 */
struct Sphere : public SingleObjectiveFunction {
	
	Sphere(std::size_t numberOfVariables = 5):m_numberOfVariables(numberOfVariables) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= HAS_FIRST_DERIVATIVE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Sphere"; }

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

	double eval(SearchPointType const& x) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		return norm_sqr(x);
	}
	
	double evalDerivative(SearchPointType const& x, FirstOrderDerivative& derivative) const {
		SIZE_CHECK(x.size() == numberOfVariables());
		m_evaluationCounter++;
		derivative.resize(x.size());
		noalias(derivative) = 2*x;
		return norm_sqr(x);
	}
private:
	std::size_t m_numberOfVariables;
};

}}

#endif
