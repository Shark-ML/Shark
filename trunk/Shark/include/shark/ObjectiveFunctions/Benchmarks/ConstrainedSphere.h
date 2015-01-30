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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CONSTRAINEDSPHERE_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CONSTRAINEDSPHERE_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/**
 * \brief Constrained Sphere function
 *
 * This is a simple sphere function minimizing \f$ f(x) = \sum_i^N x_i^2-m \f$ under the constraint that
 *  \f$ x_i \geq 1\f$ for \f$ i = 1,\dots,m \f$. The minimum is at \f$ x_1=\dots = x_m = 1\f$ and 
 * \f$ x_{m+1}=\dots = x_N = 0 \f$ with function value 0.
 *
 * This is a simple benchmark for evolutionary algorithms as, the closer the algorithm is to the optimu
 */
struct ConstrainedSphere : public SingleObjectiveFunction {
	
	ConstrainedSphere(unsigned int numberOfVariables = 5, unsigned int m = 1)
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

	void configure(const PropertyTree &node) {
		m_numberOfVariables = node.get("numberOfVariables", 5l);
		m_constraints = node.get("m", 1l);
	}

	void proposeStartingPoint(SearchPointType &x) const {
		x.resize(numberOfVariables());

		for (unsigned int i = 0; i < m_constraints; i++) {
			x(i) = std::abs(Rng::gauss(0, 1))+1;
		}
		for (unsigned int i = m_constraints; i < x.size(); i++) {
			x(i) = Rng::gauss(0, 1);
		}
	}
	
	bool isFeasible( SearchPointType const& input) const {
		for (unsigned int i = 0; i < m_constraints; i++) {
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

}

#endif
