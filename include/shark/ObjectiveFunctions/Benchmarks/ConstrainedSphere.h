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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CONSTRAINEDSPHERE_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CONSTRAINEDSPHERE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/Random.h>

namespace shark {namespace benchmarks{
/**
 * \brief Constrained Sphere function
 *
 * This is a simple sphere function minimizing \f$ f(x) = \sum_i^N x_i^2-m \f$ under the constraints that
 *  \f$ x_i \geq 1\f$ for \f$ i = 1,\dots,m \f$. The minimum is at \f$ x_1=\dots = x_m = 1\f$ and 
 * \f$ x_{m+1}=\dots = x_N = 0 \f$ with function value 0.
 *
 * This is a simple benchmark for evolutionary algorithms as, the closer the algorithm is to the optimum
* \ingroup benchmarks
 */
struct ConstrainedSphere : public SingleObjectiveFunction {
	
	ConstrainedSphere(std::size_t numberOfVariables = 5, std::size_t m = 1)
	:m_numberOfVariables(numberOfVariables), m_constraints(m) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_features |= IS_CONSTRAINED_FEATURE;
		m_features |= IS_THREAD_SAFE;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ConstrainedSphere"; }

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

		for (std::size_t i = 0; i < m_constraints; i++) {
			x(i) = std::abs(random::gauss(*mep_rng, 0, 1))+1;
		}
		for (std::size_t i = m_constraints; i < x.size(); i++) {
			x(i) = random::gauss(*mep_rng,0, 1);
		}
		return x;
	}
	
	bool isFeasible( SearchPointType const& input) const {
		for (std::size_t i = 0; i < m_constraints; i++) {
			if(input(i) < 1) return false;
		}
		return true;
	}

	double eval(const SearchPointType &p) const {
		m_evaluationCounter++;
		return norm_sqr(p)-m_constraints;
	}
private:
	std::size_t m_numberOfVariables;
	std::size_t m_constraints;
};

}}

#endif
